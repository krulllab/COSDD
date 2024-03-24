import torch
from torch import nn

from ..lib.nn import ResidualBlock


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
                 n_res_blocks,
                 n_filters,
                 is_top_layer=False,
                 downsampling_steps=None,
                 merge_type=None,
                 batchnorm=True,
                 skip=False,
                 res_block_type=None,
                 gated=None):

        super().__init__()

        self.is_top_layer = is_top_layer
        self.skip = skip

        # Downsampling steps left to undo in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block: sequence of deterministic
        # residual blocks with downsampling when needed.
        block_list = []
        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1
            block_list.append(
                TopDownResBlock(
                    n_filters,
                    n_filters,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))
        self.blocks = nn.Sequential(*block_list)

        if not is_top_layer:
            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                batchnorm=batchnorm,
                res_block_type=res_block_type,
            )

            # Skip connection that goes around the stochastic top-down layer
            if skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    merge_type='residual',
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                )

    def forward(self,
                input_=None,
                skip_connection_input=None,
                bu_value=None,):

        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

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


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """

    def __init__(self,
                 n_res_blocks,
                 n_filters,
                 downsampling_steps=0,
                 batchnorm=True,
                 res_block_type=None,
                 gated=None):
        super().__init__()

        bu_blocks = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            bu_blocks.append(
                BottomUpResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    downsample=do_resample,
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))
        self.net = nn.Sequential(*bu_blocks)

    def forward(self, x):
        return self.net(x)


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).

    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through average pooling.

    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.

    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are two
    residual block structures to choose from.
    """

    def __init__(self,
                 mode,
                 c_in,
                 c_out,
                 resample=False,
                 res_block_kernel=None,
                 groups=1,
                 batchnorm=True,
                 res_block_type=None,
                 min_inner_channels=None,
                 gated=None):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                self.pre_conv = nn.Conv2d(c_in,
                                          inner_filters,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          padding_mode='replicate',
                                          groups=groups)
            elif mode == 'top-down':  # upsample
                self.pre_conv = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(c_in, inner_filters, 1, groups=groups))
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            gated=gated,
            block_type=res_block_type,
        )

        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv2d(inner_filters, c_out, 1, groups=groups)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownResBlock(ResBlockWithResampling):

    def __init__(self, *args, upsample=False, **kwargs):
        kwargs['resample'] = upsample
        super().__init__('top-down', *args, **kwargs)


class BottomUpResBlock(ResBlockWithResampling):

    def __init__(self, *args, downsample=False, **kwargs):
        kwargs['resample'] = downsample
        super().__init__('bottom-up', *args, **kwargs)


class MergeLayer(nn.Module):
    """
    Merge two 4D input tensors by concatenating along dim=1 and passing the
    result through 1) a convolutional 1x1 layer, or 2) a residual block
    """

    def __init__(self,
                 channels,
                 merge_type,
                 batchnorm=True,
                 res_block_type=None):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        if merge_type == 'linear':
            self.layer = nn.Conv2d(channels[0] + channels[1], channels[2], 1)
        elif merge_type == 'residual':
            self.layer = nn.Sequential(
                nn.Conv2d(channels[0] + channels[1], channels[2], 1, padding=0),
                ResidualBlock(channels[2],
                              batchnorm=batchnorm,
                              block_type=res_block_type,
                              gated=True),
            )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        return self.layer(x)
