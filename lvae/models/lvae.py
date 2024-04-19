import numpy as np
import torch
from torch import nn
from ..lib.utils import crop_img_tensor, pad_img_tensor
from .lvae_layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock,
)


class LadderVAE(nn.Module):
    """
    Ladder Variational Autoencoder (LVAE) model.

    Args:
        colour_channels (int): Number of input image channels.
        img_shape (tuple): Shape of the input image (depth, height, width).
        s_code_channels (int): Number of channels in the returned latent code.
        z_dims (list(int), optional): List of dimensions for the latent variables z.
            If not provided, default value of [32] * 12 will be used.
        n_filters (int, optional): Number of filters in the convolutional layers.
            Default: 64.
        learn_top_prior (bool, optional): Whether to learn the top prior.
            Default: True.
        res_block_type (str, optional): Type of residual block. Default: "bacbac".
        merge_type (str, optional): Type of merge operation in the top-down layer.
            Default: "residual".
        stochastic_skip (bool, optional): Whether to use stochastic skip connections.
            Default: True.
        gated (bool, optional): Whether to use gated activations in the layers.
            Default: True.
        downsampling (list(list(bool)), optional): List of bool for downsampling per 
            dimension per layer. If not provided, default value of 
            [[False, False, False]] * n_layers will be used.
        mode_pred (bool, optional): Whether to predict the mode of the distribution.
            Default: False.
    """
class LadderVAE(nn.Module):


    def __init__(
        self,
        colour_channels,
        img_shape,
        s_code_channels,
        z_dims=None,
        n_filters=64,
        learn_top_prior=True,
        res_block_type="bacbac",
        merge_type="residual",
        stochastic_skip=True,
        gated=True,
        downsampling=None,
        mode_pred=False,
    ):
        if z_dims is None:
            z_dims = [32] * 12
        super().__init__()
        self.img_shape = tuple(img_shape)
        self.z_dims = z_dims
        self.n_layers = len(self.z_dims)
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.gated = gated
        self.mode_pred = mode_pred

        # Downsampling parameter should be a list with one boolean per dimension
        if downsampling is None:
            downsampling = [[False, False, False]] * self.n_layers
        elif isinstance(downsampling, bool):
            downsampling = [[downsampling] * 3] * self.n_layers
        elif all(isinstance(d, bool) for d in downsampling):
            downsampling = [[d] * 3 for d in downsampling]

        # Count overall downscale factor per dimension
        self.overall_downscale_factor_D = np.power(2, sum(d[0] for d in downsampling))
        self.overall_downscale_factor_H = np.power(2, sum(d[1] for d in downsampling))
        self.overall_downscale_factor_W = np.power(2, sum(d[2] for d in downsampling))
        self.overall_downscale_factor = (
            self.overall_downscale_factor_D,
            self.overall_downscale_factor_H,
            self.overall_downscale_factor_W,
        )

        assert len(downsampling) == self.n_layers

        # First bottom-up layer: change num channels
        self.first_bottom_up = nn.Sequential(
            nn.Conv3d(colour_channels,
                      n_filters,
                      5,
                      padding=2,
                      padding_mode="replicate"),
            nn.Mish(),
            BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
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
                    n_filters=n_filters,
                    downsampling=downsampling[i],
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
                    z_dim=z_dims[i],
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling=downsampling[i],
                    merge_type=merge_type,
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=res_block_type,
                    gated=gated,
                ))

        # Final top-down layer
        self.final_top_down = TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=s_code_channels,
                    res_block_type=res_block_type,
                    gated=gated,
                )

    def forward(self, x):
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        img_size = x.size()[2:]
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        s_code, kl = self.topdown_pass(bu_values)

        if not self.mode_pred:
            # Calculate KL divergence
            kl_sums = [torch.sum(layer) for layer in kl]
            kl_loss = sum(kl_sums) / float(
                x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        else:
            kl_loss = None

        # Restore original image size
        s_code = crop_img_tensor(s_code, img_size)

        output = {
            "kl_loss": kl_loss,
            "s_code": s_code,
        }
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
        bu_values=None,
        n_img_prior=None,
        mode_layers=None,
        constant_layers=None,
        forced_latent=None,
    ):
        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)

        # KL divergence of each layer
        kl = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, kl_elementwise = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
            )
            kl[i] = kl_elementwise

        # Final top-down layer
        out = self.final_top_down(out)

        return out, kl

    @torch.no_grad()
    def sample_from_prior(self, n_images):
        # Sample from p(z_L) and do top-down generative path
        # Spatial size of image is given by self.img_shape
        out, _ = self.topdown_pass(n_img_prior=n_images)
        generated_s_code = crop_img_tensor(out, self.img_shape)

        return generated_s_code

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
        Returns the smallest size (D, H, W) of the image with actual size given
        as input, such that D, H and W are powers of 2.
        :param size: input size, tuple either (N, C, D, H, w) or (D, H, W)
        :return: 3-tuple (D, H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (depth, heigth, width)
        if len(size) == 5:
            size = size[2:]
        if len(size) != 3:
            msg = ("input size must be either (N, C, D, H, W) or (D, H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // d + 1) * d for d, s in zip(dwnsc, size))

        return padded_size

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        d = sz[0] // dwnsc[0]
        h = sz[1] // dwnsc[1]
        w = sz[2] // dwnsc[2]
        c = self.z_dims[-1] * 2  # mu and log-sigma
        top_layer_shape = (n_imgs, c, d, h, w)
        return top_layer_shape
