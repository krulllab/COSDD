import numpy as np
from torch import nn
from torch.nn import functional as F
from ..lib.utils import crop_img_tensor, pad_img_tensor
from .unet_layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownResBlock,
    BottomUpResBlock,
)


class UNet(nn.Module):
    """UNet model. Deterministic version of LVAE.
    Args:
        colour_channels (int): Number of colour channels in the input image.
        blocks_per_layer (int): Number of residual blocks per layer.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the UNet.
        res_block_type (str): Type of residual block. Default: 'bacbac'.
        merge_type (str): Type of merge layer. Default: 'residual'.
        td_skip (bool): Whether to use skip connections in the top-down pass.
        gated (bool): Whether to use gated activations in the residual blocks.
        batchnorm (bool): Whether to use batch normalization in the residual blocks.
        downsampling (list): Number of downsampling steps per layer.
        loss_fn (str): Loss function to use. Default: 'L2'.
    """

    def __init__(
        self,
        colour_channels,
        blocks_per_layer=1,
        n_filters=64,
        n_layers=14,
        res_block_type="bacbac",
        merge_type="residual",
        td_skip=True,
        gated=True,
        batchnorm=True,
        downsampling=None,
        loss_fn="L2",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.td_skip = td_skip
        self.gated = gated
        self.loss_fn = loss_fn

        # We need to optimize the s_decoder separately
        # from the main VAE and noise_model
        self.automatic_optimization = False

        # Number of downsampling steps per layer
        if downsampling is None:
            downsampling = [0] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(downsampling))

        assert max(downsampling) <= self.blocks_per_layer
        assert len(downsampling) == self.n_layers

        # First bottom-up layer: change num channels
        self.first_bottom_up = nn.Sequential(
            nn.Conv2d(colour_channels,
                      n_filters,
                      5,
                      padding=2,
                      padding_mode="replicate"),
            nn.Mish(),
            BottomUpResBlock(
                c_in=n_filters,
                c_out=n_filters,
                batchnorm=batchnorm,
                res_block_type=res_block_type,
            ),
        )

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))

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
                TopDownLayer(
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=downsampling[i],
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    skip=td_skip,
                    res_block_type=res_block_type,
                    gated=gated,
                ))

        # Final top-down layer
        modules = list()
        for i in range(blocks_per_layer):
            modules.append(
                TopDownResBlock(
                    c_in=n_filters,
                    c_out=n_filters if i <
                    (blocks_per_layer - 1) else colour_channels,
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))
        self.final_top_down = nn.Sequential(*modules)

    def forward(self, x):
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        img_size = x.size()[2:]
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        output = self.topdown_pass(bu_values)

        # Restore original image size
        output = crop_img_tensor(output, img_size)

        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(
        self,
        bu_values,
    ):

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                bu_value=bu_value,
            )

        # Final top-down layer
        out = self.final_top_down(out)

        return out

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x

    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size

    def loss(self, x, y):
        if self.loss_fn == "L1":
            return F.l1_loss(x, y, reduction="none")
        elif self.loss_fn == "L2":
            return F.mse_loss(x, y, reduction="none")
