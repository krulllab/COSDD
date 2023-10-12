import numpy as np
import torch
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from ..lib.utils import crop_img_tensor, pad_img_tensor
from .lvae_layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock,
)


class LadderVAE(nn.Module):
    """Hierarchical variational autoencoder.

    Parameters
    ----------
    colour_channels : int
        Number of colour channels in input.
    img_shape : tuple
        Spatial dimensions of the input (Height, Width)
    s_code_channels : int
        Numer of channels in latent code.
    z_dims : list
        Number of feature channels at each layer of the hierarchy.
    blocks_per_layer : int
        Number of residual blocks between each latent.
    n_filters : int
        Numer of feature channels.
    learn_top_prior : bool
        Whether to learn the parameters of topmost prior.
    res_block_type : string
        The ordering of operations within each block. See ..lib.nn.ResidualBlock
    merge_type : string
        How features from bottom-up pass will be merged with features from top-down pass. See .lvae_layers.MergeLayer
    stochastic_skip : bool
        Whether to use skip connections from previous layer of hierarchy.
    gated : bool
        Whether to uses forget gate activation.
    batchnorm : bool
        Use of batch normalisation.
    downsample : list
        Number of times to downsample for each latent variable.
    mode_pred : bool
        If false, losses will not be calculated.
    """    

    def __init__(
        self,
        colour_channels,
        img_shape,
        s_code_channels,
        z_dims=None,
        blocks_per_layer=1,
        n_filters=64,
        learn_top_prior=True,
        res_block_type="bacbac",
        merge_type="residual",
        stochastic_skip=True,
        gated=True,
        batchnorm=True,
        downsampling=None,
        mode_pred=False,
    ):
        if z_dims is None:
            z_dims = [32] * 12
        super().__init__()
        self.img_shape = tuple(img_shape)
        self.z_dims = z_dims
        self.n_layers = len(self.z_dims)
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.gated = gated
        self.mode_pred = mode_pred

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
            BottomUpDeterministicResBlock(
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
                    z_dim=z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=downsampling[i],
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=res_block_type,
                    gated=gated,
                ))

        # Final top-down layer
        modules = list()
        for i in range(blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters if i <
                    (blocks_per_layer - 1) else s_code_channels,
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
        s_code, kl = self.topdown_pass(bu_values)

        if not self.mode_pred:
            # Calculate KL divergence
            kl_sums = [torch.sum(layer) for layer in kl]
            kl_loss = sum(kl_sums) / float(
                x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
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
        generated_s = self.s_decoder(generated_s_code)
        generated_x = self.noise_model.sample(generated_s_code)

        return generated_s, generated_x

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

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and log-sigma
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape
