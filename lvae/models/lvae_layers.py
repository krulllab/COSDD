import torch
from torch import nn

from ..lib.nn import ResidualBlock
from ..lib.stochastic import NormalStochasticBlock3d


class TopDownLayer(nn.Module):
    """
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.

    The architecture when doing inference is roughly as follows:
       p_params = output of top-down layer above
       bu = inferred bottom-up value at this layer
       q_params = merge(bu, p_params)
       z = stochastic_layer(q_params)
       possibly get skip connection from previous top-down layer
       top-down deterministic ResNet

    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.

    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.
    """

    def __init__(self,
                 z_dim,
                 n_filters,
                 is_top_layer=False,
                 upsampling=False,
                 merge_type=None,
                 kernel_size=3,
                 stochastic_skip=False,
                 res_block_type=None,
                 gated=None,
                 learn_top_prior=False,
                 top_prior_param_shape=None):

        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip

        # Define top layer prior parameters
        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_shape),
                requires_grad=learn_top_prior)

        # Define deterministic top-down residual block
        self.res_block = TopDownDeterministicResBlock(
                    n_filters,
                    n_filters,
                    upsampling=upsampling,
                    res_block_kernel=kernel_size,
                    res_block_type=res_block_type,
                    gated=gated,
                )

        # Define stochastic block with 3d convolutions
        self.stochastic = NormalStochasticBlock3d(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            transform_p_params=(not is_top_layer),
        )

        if not is_top_layer:
            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                res_block_type=res_block_type,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    merge_type='residual',
                    res_block_type=res_block_type,
                )

    def forward(self,
                input_=None,
                skip_connection_input=None,
                inference_mode=False,
                bu_value=None,
                n_img_prior=None,
                forced_latent=None,
                use_mode=False,
                force_constant_output=False,
                mode_pred=False):

        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1, -1)

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
            else:
                q_params = self.merge(bu_value, p_params)

        # In generative mode, q is not used
        else:
            q_params = None

        # Sample from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        z, kl_elementwise = self.stochastic(
            p_params=p_params,
            q_params=q_params,
            forced_latent=forced_latent,
            use_mode=use_mode,
            force_constant_output=force_constant_output,
            mode_pred=mode_pred)

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            z = self.skip_connection_merger(z, skip_connection_input)

        # Last top-down block (sequence of residual blocks)
        z = self.res_block(z)

        return z, kl_elementwise


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference. Essentially a ResBlock
    with optional per-dimension downsampling.
    """

    def __init__(self,
                 n_filters,
                 downsampling=False,
                 kernel_size=3,
                 res_block_type=None,
                 gated=None):
        super().__init__()

        self.res_block = BottomUpDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    downsampling=downsampling,
                    res_block_kernel=kernel_size,
                    res_block_type=res_block_type,
                    gated=gated,
                )

    def forward(self, x):
        return self.res_block(x)


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).

    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.

    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.

    Other parameters: kernel size, and groups of the internal
    residual block; whether the residual path has a gate layer at the end. 
    """

    def __init__(self,
                 mode,
                 c_in,
                 c_out,
                 resample=False,
                 res_block_kernel=None,
                 groups=1,
                 res_block_type=None,
                 min_inner_channels=None,
                 gated=None):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        if isinstance(resample, bool):
            resample = [resample] * 3
        else:
            assert len(resample) == 3
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                stride = tuple([int(x) + 1 for x in resample])
                self.pre_conv = nn.Conv3d(c_in,
                                          inner_filters,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          padding_mode='replicate',
                                          groups=groups)
            elif mode == 'top-down':  # upsample
                scale_factor = tuple([int(x) + 1 for x in resample])
                self.pre_conv = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
                    nn.Conv3d(c_in, inner_filters, 1, groups=groups))
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv3d(c_in, inner_filters, 1, groups=groups)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            kernel_size=res_block_kernel,
            gated=gated,
            block_type=res_block_type,
        )

        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv3d(inner_filters, c_out, 1, groups=groups)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, upsampling=False, **kwargs):
        kwargs['resample'] = upsampling
        super().__init__('top-down', *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, downsampling=False, **kwargs):
        kwargs['resample'] = downsampling
        super().__init__('bottom-up', *args, **kwargs)


class MergeLayer(nn.Module):
    """
    Merge two 4D input tensors by concatenating along dim=1 and passing the
    result through 1) a convolutional 1x1 layer, or 2) a residual block
    """

    def __init__(self,
                 channels,
                 merge_type,
                 kernel_size=3,
                 res_block_type=None):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels] * 3
        assert len(channels) == 3

        if merge_type == 'linear':
            self.layer = nn.Conv3d(channels[0] + channels[1], channels[2], 1)
        elif merge_type == 'residual':
            self.layer = nn.Sequential(
                nn.Conv3d(channels[0] + channels[1], channels[2], 1, padding=0),
                ResidualBlock(channels[2],
                              gated=True,
                              kernel_size=kernel_size,
                              block_type=res_block_type),
            )

    def forward(self, x, y):
        if x.shape[2:] != y.shape[2:]:
            raise ValueError(f"Spatial dimensions do not match. x: {x.shape}, y: {y.shape}")

        x = torch.cat((x, y), dim=1)
        return self.layer(x)
