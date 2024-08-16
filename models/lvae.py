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
from torch.utils.checkpoint import checkpoint
from torch.distributions import kl_divergence

from .utils import get_padded_size, spatial_pad_crop
from .nn import Conv, ResBlockWithResampling, BottomUpLayer, VAETopDownLayer


class LadderVAE(nn.Module):
    """
    Ladder Variational Autoencoder (LVAE) model.

    Args:
        colour_channels (int): Number of input image channels.
        img_size (tuple): size of the input image (height, width).
        s_code_channels (int): Number of channels in the returned latent code.
        z_dims (list(int)): List of dimensions for the latent variables z.
        blocks_per_layer (int): Number of residual blocks per layer.
        n_filters (int): Number of filters in the convolutional layers.
        learn_top_prior (bool): Whether to learn the top prior.
        stochastic_skip (bool): Whether to use stochastic skip connections.
        downsampling (list(int)): Binary list of downsampling per layer.
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        colour_channels,
        img_size,
        s_code_channels,
        z_dims=None,
        blocks_per_layer=1,
        n_filters=64,
        learn_top_prior=True,
        stochastic_skip=True,
        downsampling=None,
        checkpointed=False,
        dimensions=2,
    ):
        if z_dims is None:
            z_dims = [32] * 12
        super().__init__()
        self.img_size = tuple(img_size)
        self.z_dims = z_dims
        self.n_layers = len(self.z_dims)
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.checkpointed = checkpointed

        # Number of downsampling steps per layer
        if downsampling is None:
            downsampling = [0] * self.n_layers
        self.n_downsample = sum(downsampling)


        assert max(downsampling) <= self.blocks_per_layer
        assert len(downsampling) == self.n_layers

        self.first_bottom_up = nn.Sequential(
            Conv(
                colour_channels,
                n_filters,
                5,
                padding=2,
                padding_mode="replicate",
                dimensions=dimensions,
            ),
            nn.Mish(),
            ResBlockWithResampling(
                c_in=n_filters,
                c_out=n_filters,
                gated=False,
                dimensions=dimensions,
            ),
        )

        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    dimensions=dimensions,
                )
            )

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            self.top_down_layers.append(
                VAETopDownLayer(
                    z_dim=z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling_steps=downsampling[i],
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_size=self.get_top_prior_param_size(),
                    dimensions=dimensions,
                )
            )

        self.final_top_down = nn.Sequential()
        for i in range(blocks_per_layer):
            self.final_top_down.append(
                ResBlockWithResampling(
                    c_in=n_filters,
                    c_out=n_filters if i < (blocks_per_layer - 1) else s_code_channels,
                    dimensions=dimensions,
                )
            )

    def forward(self, x):
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        x_size = x.size()[2:]
        padded_size = get_padded_size(x_size, self.n_downsample)
        x = spatial_pad_crop(x, padded_size)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x)

        z, q_list, p_list = self.topdown_pass(bu_values)

        # Restore original image size
        z = spatial_pad_crop(z, x_size)

        output = {
            "q_list": q_list,
            "p_list": p_list,
            "s_code": z,
        }
        return output

    def bottomup_pass(self, x):
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            if i % 2 == 0 and self.checkpointed:
                x = checkpoint(
                    self.bottom_up_layers[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(
        self,
        bu_values=None,
        n_img_prior=None,
        mode_layers=None,
        constant_layers=None,
        forced_latent=None,
    ):
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        if bu_values is None:
            bu_values = [None] * self.n_layers

        # For the KL divergence of each layer
        q_list = []
        p_list = []

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        p_params = None
        for i in reversed(range(self.n_layers)):
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            skip_input = p_params

            if i % 2 == 0 and self.checkpointed:
                z, q, p = checkpoint(
                    self.top_down_layers[i],
                    p_params,
                    skip_input,
                    bu_values[i],
                    n_img_prior,
                    forced_latent[i],
                    use_mode,
                    constant_out,
                    use_reentrant=False,
                )
            else:
                z, q, p = self.top_down_layers[i](
                    p_params,
                    skip_input,
                    bu_values[i],
                    n_img_prior,
                    forced_latent[i],
                    use_mode,
                    constant_out,
                )
            p_params = z
            q_list.append(q)
            p_list.append(p)

        z = self.final_top_down(z)

        return z, q_list, p_list

    def get_top_prior_param_size(self):
        padded_size = get_padded_size(self.img_size, self.n_downsample)
        dwnsc = 2**self.n_downsample
        top_prior_size = [s // dwnsc for s in padded_size]
        c = self.z_dims[-1] * 2  # mu and log-sigma
        top_prior_size = [1, c] + top_prior_size

        return top_prior_size

    @torch.no_grad()
    def sample_from_prior(self, n_images):
        z, _, _ = self.topdown_pass(n_img_prior=n_images)
        generated_s_code = spatial_pad_crop(z, self.img_size)

        return generated_s_code

    @staticmethod
    def kl_divergence(q_list, p_list):
        kl_sum = 0
        for q, p in zip(q_list, p_list):
            kl_sum = kl_sum + kl_divergence(q, p).sum()
        return kl_sum
