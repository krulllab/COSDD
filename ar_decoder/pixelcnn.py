import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, MixtureSameFamily
from tqdm import tqdm

from .nn import Layers
from .lib import sample_mixture_model


class PixelCNN(nn.Module):
    """Autoregressive decoder.

    Causal convolutions are one dimensional, with shape (1, kernel_size).
    To implement veritcal kernels, input is rotated 90 degrees on entry and
    rotated back on exit.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        s_code_channels (int): Number of channels in the decoded signal code.
        kernel_size (int): Size of the kernel in the convolutional layers.
        RF_shape (str): Orientation of the receptive field. Can be "horizontal" or "vertical".
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers.
        n_gaussians (int): Number of gaussians in the predictive mixture model.

    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        kernel_size,
        RF_shape="horizontal",
        n_filters=64,
        n_layers=4,
        n_gaussians=5,
    ):
        super().__init__()
        assert RF_shape in ("horizontal", "vertical")
        self.n_gaussians = n_gaussians
        self.colour_channels = colour_channels
        self.RF_shape = RF_shape

        if RF_shape == "horizontal":
            rotate90 = False
        elif RF_shape == "vertical":
            rotate90 = True

        self.layers = Layers(
            colour_channels=colour_channels,
            s_code_channels=s_code_channels,
            kernel_size=kernel_size,
            n_filters=n_filters,
            n_layers=n_layers,
            rotate90=rotate90,
        )

        c_out = n_gaussians * colour_channels * 3
        self.out_conv = nn.Conv2d(
            in_channels=n_filters, out_channels=c_out, kernel_size=1
        )

    def forward(self, x, s_code):
        out = self.layers(x, s_code)

        return self.out_conv(out)

    def extract_params(self, params):
        logweights = params[:, 0::3].unfold(1, self.n_gaussians, self.n_gaussians)
        loc = params[:, 1::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = params[:, 2::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = nn.functional.softplus(scale)
        return logweights, loc, scale

    def loglikelihood(self, x, params):
        logweights, loc, scale = self.extract_params(params)
        
        p = MixtureSameFamily(
            Categorical(logits=logweights),
            Normal(loc=loc, scale=scale)
        )

        return p.log_prob(x)

    @torch.no_grad()
    def sample(self, s_code):
        image = torch.zeros(
            s_code.shape[0], self.colour_channels, s_code.shape[2], s_code.shape[3]
        )
        image = image.to(s_code.get_device())

        if self.RF_shape == "horizontal":
            for i in tqdm(range(s_code.shape[-1]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1], s_code[..., : i + 1])
                logweights, loc, scale = self.extract_params(params[..., i : i + 1])
                image[..., i : i + 1] = sample_mixture_model(logweights, loc, scale)
        else:
            for i in tqdm(range(s_code.shape[-2]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1, :], s_code[..., : i + 1, :])
                logweights, loc, scale = self.extract_params(params[..., i : i + 1, :])
                image[..., i : i + 1, :] = sample_mixture_model(logweights, loc, scale)
        
        return image
