"""
MIT License

Copyright (c) 2019 Andrea Dittadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.distributions import Normal

from .utils import Rotate90, interleave, LinearUpsample


class Conv(nn.Module):
    """
    Convolutional layer for 1, 2, or 3D data.

    Same arugments as nn.Convxd but dimensionality of data is
    passed as argument.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        dimensions=2,
    ):
        super().__init__()
        self.args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype,
        }
        conv = getattr(nn, f"Conv{dimensions}d")
        self.conv = conv(**self.args)

    def forward(self, x):
        return self.conv(x)


class GateLayer(nn.Module):
    def __init__(self, channels, kernel_size, dimensions=2):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = Conv(channels, 2 * channels, kernel_size, padding=pad, dimensions=dimensions)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x, gate = x[:, 0::2], x[:, 1::2]
        x = self.nonlin(x)
        gate = torch.sigmoid(gate)
        return x * gate


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        groups=1,
        gated=True,
        dimensions=2,
    ):
        super().__init__()
        self.channels = channels
        assert kernel_size % 2 == 1
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size
        self.groups = groups
        self.gated = gated

        BatchNorm = getattr(nn, f"BatchNorm{dimensions}d")

        self.block = nn.Sequential()
        for _ in range(2):
            self.block.append(BatchNorm(self.channels))
            self.block.append(nn.Mish())
            conv = Conv(
                self.channels,
                self.channels,
                self.kernel_size,
                padding=self.pad,
                groups=self.groups,
                dimensions=dimensions,
            )
            self.block.append(conv)
        if self.gated:
            self.block.append(GateLayer(self.channels, 1, dimensions=dimensions))

    def forward(self, x):
        return self.block(x) + x
    

class ResBlockWithResampling(nn.Module):
    """
    Residual Block with Resampling.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        resample (str): Resampling method. Can be "up", "down", or None.
        res_block_kernel (int): Kernel size for the residual block.
        groups (int): Number of groups for grouped convolution.
        gated (bool): Whether to use gated activation.
        dimensions (int): Dimensionality of the data (1, 2 or 3).

    Attributes:
        pre_conv (nn.Module): Pre-convolutional layer.
        res_block (ResidualBlock): Residual block layer.

    """

    def __init__(
        self,
        c_in,
        c_out,
        resample=None,
        res_block_kernel=3,
        groups=1,
        gated=True,
        dimensions=2,
    ):
        super().__init__()
        assert resample in ["up", "down", None]

        if resample == "up":
            self.pre_conv = nn.Sequential(
                LinearUpsample(scale_factor=2),
                Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions),
            )
        elif resample == "down":
            self.pre_conv = Conv(
                c_in,
                c_out,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
                groups=groups,
                dimensions=dimensions,
            )
        elif c_in != c_out:
            self.pre_conv = Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions)
        else:
            self.pre_conv = nn.Identity()

        self.res_block = ResidualBlock(
            channels=c_out,
            kernel_size=res_block_kernel,
            groups=groups,
            gated=gated,
            dimensions=dimensions,
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res_block(x)
        return x


class MergeLayer(nn.Module):
    """
    A module that merges two input tensors and applies a convolutional layer followed by a residual block.

    Args:
        channels (int or list[int]): The number of input channels for the convolutional layer and the residual block.
            If an integer is provided, it will be used for all three channels. If a list of integers is provided,
            it should have a length of 3, representing the number of channels for each input.
        dimensions (int): Dimensionality of the data (1, 2 or 3)

    Attributes:
        layer (nn.Sequential): The sequential module that consists of a convolutional layer and a residual block.

    """

    def __init__(self, channels, dimensions=2):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        self.layer = nn.Sequential(
            Conv(channels[0] + channels[1], channels[2], 1, dimensions=dimensions),
            ResidualBlock(
                channels[2],
                dimensions=dimensions,
            ),
        )

    def forward(self, x, y):
        x = interleave(x, y)
        return self.layer(x)


class BottomUpLayer(nn.Module):
    """
    Module that consists of multiple residual blocks.
    Each residual block can optionally perform downsampling.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        downsampling_steps (int, optional): The number of downsampling steps to perform. Defaults to 0.
        dimensions (int): Dimensionality of the data (1, 2 or 3)

    Attributes:
        bu_blocks (nn.Sequential): Sequential container for the residual blocks.

    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        downsampling_steps=0,
        dimensions=2,
    ):
        super().__init__()

        self.bu_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            resample = None
            if downsampling_steps > 0:
                resample = "down"
                downsampling_steps -= 1
            self.bu_blocks.append(
                ResBlockWithResampling(
                    c_in=n_filters,
                    c_out=n_filters,
                    resample=resample,
                    dimensions=dimensions,
                )
            )

    def forward(self, x):
        return self.bu_blocks(x)


class TopDownLayer(nn.Module):
    """
    Module that consists of multiple residual blocks.
    Each residual block can optionally perform upsampling.

    Merges a bottom-up skip connection and optionally a skip connection from the previous layer.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        is_top_layer (bool): Whether the layer is the top layer.
        upsampling_steps (int, optional): The number of downsampling steps to perform. Defaults to 0.
        skip (bool, optional): Whether to use a skip connection from the previous layer. Defaults to False.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
        
    Attributes:
        blocks (nn.Sequential): Sequential container for the residual blocks.
        merge (MergeLayer): Merge layer for combining the bottom-up and top-down paths.
        skip_connection_merger (MergeLayer): Merge layer for combining the skip connection and the output of the residual blocks.

    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        upsampling_steps=None,
        skip=False,
        dimensions=2,
    ):
        super().__init__()

        self.is_top_layer = is_top_layer
        self.skip = skip

        self.blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            resample = None
            if upsampling_steps > 0:
                resample = "up"
                upsampling_steps -= 1
            self.blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample=resample,
                    dimensions=dimensions,
                )
            )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                dimensions=dimensions,
            )

            if skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    dimensions=dimensions,
                )

    def forward(
        self,
        input_=None,
        skip_connection_input=None,
        bu_value=None,
    ):
        if self.is_top_layer:
            output = bu_value
        else:
            output = self.merge(bu_value, input_)

        # Skip connection from previous layer
        if self.skip and not self.is_top_layer:
            output = self.skip_connection_merger(output, skip_connection_input)

        # Last top-down block (sequence of residual blocks)
        output = self.blocks(output)

        return output


### Ladder VAE parts
class NormalStochasticBlock(nn.Module):
    """
    Stochastic layer for the inference/generation path of a Ladder VAE.

    Creates p(z{i}|z{i+1}) (and q(z{i}|z{i+1}, x)) for generation (or inference).

    Args:
        c_in (int): Number of input channels.
        c_vars (int): Number of latent variable channels.
        c_out (int): Number of output channels.
        kernel (int): Kernel size for the convolutional layers.
        transform_p_params (bool): Whether to double the channels of p(z) with a convolutional layer.
        dimensions (int): Dimensionality of the data (1, 2 or 3)

    """

    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True, dimensions=2):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = Conv(
                c_in,
                2 * c_vars,
                kernel,
                padding=pad,
                padding_mode="replicate",
                dimensions=dimensions,
            )
        self.conv_in_q = Conv(
            c_in,
            2 * c_vars,
            kernel,
            padding=pad,
            padding_mode="replicate",
            dimensions=dimensions,
        )
        self.conv_out = Conv(
            c_vars,
            c_out,
            kernel,
            padding=pad,
            padding_mode="replicate",
            dimensions=dimensions,
        )

    def forward(
        self,
        p_params,
        q_params=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
    ):
        """
        Forward pass of the stochastic layer.

        Args:
            p_params (torch.Tensor): Parameters of the prior distribution p(z).
            q_params (torch.Tensor, optional): Parameters of the approximate posterior distribution q(z|x).
            forced_latent (torch.Tensor, optional): Forced latent variable.
            use_mode (bool, optional): Whether to use the mode of the distribution.
            force_constant_output (bool, optional): Whether to force the output to be constant across batch.

        Returns:
            torch.Tensor: Sample from either q(z|x) or p(z).
            torch.Distribution: q(z|x)
            torch.Distribution: p(z)
        """

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_std_ = p_params[:, 0::2], p_params[:, 1::2]
        p_std = nn.functional.softplus(p_std_)
        p = Normal(p_mu, p_std, validate_args=True)

        if q_params is not None:
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_std_ = q_params[:, 0::2], q_params[:, 1::2]
            q_std = nn.functional.softplus(q_std_)
            q = Normal(q_mu, q_std, validate_args=True)

            # Sample from q(z)
            sampling_distrib = q
        else:
            # Sample from p(z)
            q = None
            sampling_distrib = p

        # Generate latent variable (typically by sampling)
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                z = sampling_distrib.rsample()
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()

        # Output of stochastic layer
        z = self.conv_out(z)

        return z, q, p


class VAETopDownLayer(nn.Module):
    """
    Top-down layer for the generative path of a Ladder VAE.

    Merges a bottom-up skip connection for the approximate posterior
    and optionally a skip connection from the previous layer.

    Args:
        z_dim (int): Channels of the latent variable.
        n_res_blocks (int): Number of residual blocks in the layer.
        n_filters (int): Number of filters in each residual block.
        is_top_layer (bool): Whether the layer is the top layer.
        upsampling_steps (int): The number of upsampling steps to perform.
        stochastic_skip (bool): Whether to include a skip connection around the stochastic layer.
        learn_top_prior (bool): Whether to learn the parameters of the top prior.
        top_prior_param_size (int): Spatial size of the top prior parameters.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        z_dim,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        upsampling_steps=None,
        stochastic_skip=False,
        learn_top_prior=False,
        top_prior_param_size=None,
        dimensions=2,
    ):
        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip

        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_size), requires_grad=learn_top_prior
            )

        self.deterministic_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            resample = None
            if upsampling_steps > 0:
                resample = "up"
                upsampling_steps -= 1
            self.deterministic_blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample=resample,
                    dimensions=dimensions,
                )
            )

        self.stochastic = NormalStochasticBlock(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            transform_p_params=(not is_top_layer),
            dimensions=dimensions,
        )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                dimensions=dimensions,
            )

            if stochastic_skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    dimensions=dimensions,
                )

    def forward(
        self,
        p_params=None,
        skip_connection_input=None,
        bu_value=None,
        n_img_prior=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
    ):
        """
        Forward pass of the top-down layer.

        Args:
            p_params (torch.Tensor, optional): Parameters of the prior distribution p(z).
            skip_connection_input (torch.Tensor, optional): Skip connection from the previous layer.
            bu_value (torch.Tensor, optional): Inferred bottom-up value at this layer.
            n_img_prior (int, optional): Number of images to generate.
            forced_latent (torch.Tensor, optional): Forced latent variable.
            use_mode (bool, optional): Whether to use the mode of the distribution.
            force_constant_output (bool, optional): Whether to force the output to be constant across batch.

        Returns:
            torch.Tensor: Sample from either q(z|x) or p(z).
            torch.Distribution: q(z|x)
            torch.Distribution: p(z)
        """

        if self.is_top_layer:
            assert p_params is None
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.repeat_interleave(n_img_prior, dim=0)

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if bu_value is not None:
            if self.is_top_layer:
                q_params = bu_value
            else:
                q_params = self.merge(bu_value, p_params)
        # In generative mode, q is not used
        else:
            q_params = None

        # Samples from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        z, q, p = self.stochastic(
            p_params=p_params,
            q_params=q_params,
            forced_latent=forced_latent,
            use_mode=use_mode,
            force_constant_output=force_constant_output,
        )

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            z = self.skip_connection_merger(z, skip_connection_input)

        z = self.deterministic_blocks(z)

        return z, q, p


### AR Decoder (PixelCNN) Parts
class ShiftedConv(nn.Module):
    """
    Convolutional layer with receptive field shifted left on the last dimension.

    Can be applied to 1, 2, or 3D tensors, with parameters initialised by
    passing a tensor of size (N, C, *dims) through the layer.
    The kernel spans only the last dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): Dilation rate of the convolutional kernel. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        first (bool, optional): Whether this is the first layer in the network. Default is False.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
        first=False,
        dimensions=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.first = first

        shift = dilation * (kernel_size - 1)
        self.padding = (shift, 0)

        kernel_size = [1] * (dimensions - 1) + [self.kernel_size]
        self.conv = Conv(
            self.in_channels,
            self.out_channels,
            kernel_size,
            dilation=self.dilation,
            groups=self.groups,
            dimensions=dimensions,
        )

        kernel_mask = torch.ones((1, 1, *kernel_size))
        if self.first:
            kernel_mask[..., -1] = 0
        self.register_buffer("kernel_mask", kernel_mask)

    def forward(self, x):
        if not hasattr(self, "conv"):
            self.lazy_dims_init(x)
        x = F.pad(x, self.padding)
        self.conv.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class PixelCNNBlock(nn.Module):
    """
    Residual block for autoregressive CNN.

    Uses ShiftedConv layers for the main convolutional layers.
    Conditions on a tensor with same spatial shape as the input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        s_code_channels (int): Number of channels in the s_code.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): Dilation rate of the convolutional kernel. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        first (bool, optional): Whether this is the first layer in the network so should mask centre pixel. Default is False.
        dimensions (int): Dimensionality of the data (1, 2 or 3)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        s_code_channels,
        kernel_size,
        dilation=1,
        groups=1,
        gated=False,
        first=False,
        dimensions=2,
    ):
        super().__init__()
        mid_channels = out_channels * 2 if gated else out_channels

        self.in_conv = ShiftedConv(
            in_channels,
            mid_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
            first=first,
            dimensions=dimensions,
        )
        self.s_conv = Conv(s_code_channels, mid_channels, 1, dimensions=dimensions)
        if gated:
            self.act_fn = lambda x: torch.tanh(x[:, 0::2]) * torch.sigmoid(x[:, 1::2])
        else:
            self.act_fn = nn.ReLU()
        self.out_conv = Conv(out_channels, out_channels, 1, groups=groups, dimensions=dimensions)

        self.do_skip = out_channels == in_channels and not first

    def forward(self, x, s_code):
        """
        Forward pass of the PixelCNN block.

        Returns condiontal tensor to match other layers used in the PixelCNN.

        Args:
            x (torch.Tensor): Input tensor.
            s_code (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Condition tensor.
        """
        feat = self.in_conv(x) + self.s_conv(s_code)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out, s_code


class PixelCNNLayers(nn.Module):
    """
    Layers for autoregressive CNN.

    Used to model spatially correlated noise in the input tensor.
    Noise can be correlated along either x, y or z axis. Rotates input
    tensor to align noise with the x axis for simplicity.
    Assumes noise is not correlated along the channel axis by using
    grouped convolutions.

    Args:
        colour_channels (int): Number of input channels.
        s_code_channels (int): Number of channels in the conditioning tensor.
        kernel_size (int): Size of the convolutional kernel.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the PixelCNN.
        direction (str): Axis along which noise is correlated. Default is 'x'.
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        kernel_size,
        n_filters,
        n_layers,
        direction="x",
        gated=False,
        checkpointed=False,
        dimensions=2,
    ):
        super().__init__()
        self.checkpointed = checkpointed

        middle_layer = n_layers // 2 + n_layers % 2

        self.layers = nn.ModuleList()
        if direction == "y":
            self.layers.append(Rotate90(k=1, dims=[-2, -1]))
        elif direction == "z":
            self.layers.append(Rotate90(k=1, dims=[-3, -1]))
        elif direction not in ("x", "t"):
            raise ValueError(
                f"Direction {direction} not supported. Use 't', 'x', 'y' or 'z'."
            )

        groups = colour_channels
        for i in range(n_layers):
            c_in = colour_channels if i == 0 else n_filters
            first = i == 0
            dilation = 2**i if i < middle_layer else 2 ** (n_layers - i - 1)

            self.layers.append(
                PixelCNNBlock(
                    in_channels=c_in,
                    out_channels=n_filters,
                    s_code_channels=s_code_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    gated=gated,
                    first=first,
                    dimensions=dimensions,
                )
            )
            self.layers.append(
                PixelCNNBlock(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    s_code_channels=s_code_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    gated=gated,
                    dimensions=dimensions,
                )
            )

        if direction == "y":
            self.layers.append(Rotate90(k=-1, dims=[-2, -1]))
        elif direction == "z":
            self.layers.append(Rotate90(k=-1, dims=[-3, -1]))

    def forward(self, x, s_code):
        """
        Forward pass of the PixelCNN layers.

        Args:
            x (torch.Tensor): Input tensor.
            s_code (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.checkpointed:
                x, s_code = checkpoint(
                    layer,
                    x, 
                    s_code,
                    use_reentrant=False,
                )
            else:
                x, s_code = layer(x, s_code)
        return x
