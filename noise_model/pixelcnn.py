import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm

from .nn import Layers, OutConvs
from .lib import sample_mixture_model


class PixelCNN(nn.Module):
    """Autoregressive decoder.

    Causal convolutions are one dimensional, with shape (1, kernel_size).
    To implement veritcal kernels, input is rotated 90 degrees on entry and
    rotated back on exit.

    Parameters
    ----------
    colour_channels : int
        Number of colour channels in the image.
    s_code_channels : int
        Number of channels in the latent code.
    kernel_size : int
        Size of the convolutional kernels.
    RF_shape : str, optional
        Shape of the receptive field. Either 'horizontal' or 'vertical'.
    n_filters : int, optional
        Number of filters in the convolutional layers.
    n_layers : int, optional
        Number of convolutional layers.
    n_out_layers : int, optional
        Number of convolutional layers in the output.
    n_gaussians : int, optional
        Number of gaussians in the mixture model.

    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        kernel_size,
        RF_shape="horizontal",
        n_filters=64,
        n_layers=4,
        n_out_layers=3,
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
        self.out_convs = OutConvs(
            in_channels=n_filters, out_channels=c_out, n_layers=n_out_layers
        )

    def forward(self, x, s_code):
        out = self.layers(x, s_code)

        return self.out_convs(out)

    def extract_params(self, params):
        weights = params[:, 0::3].unfold(1, self.n_gaussians, self.n_gaussians)
        weights = nn.functional.softmax(weights, dim=-1)

        loc = params[:, 1::3].unfold(1, self.n_gaussians, self.n_gaussians)

        scale = params[:, 2::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = nn.functional.softplus(scale)
        return weights, loc, scale

    def loglikelihood(self, x, params):
        weights, loc, scale = self.extract_params(params)

        loglikelihoods = Normal(loc, scale, validate_args=False).log_prob(x[..., None])
        temp = loglikelihoods.max(dim=-1, keepdim=True)[0]
        loglikelihoods = loglikelihoods - temp
        loglikelihoods = loglikelihoods.exp()
        loglikelihoods = loglikelihoods * weights
        loglikelihoods = loglikelihoods.sum(dim=-1, keepdim=True)
        loglikelihoods = loglikelihoods.log()
        loglikelihoods = loglikelihoods + temp

        return loglikelihoods

    @torch.no_grad()
    def sample(self, s_code):
        image = torch.zeros(
            s_code.shape[0], self.colour_channels, s_code.shape[2], s_code.shape[3]
        )
        image = image.to(s_code.get_device())

        for i in tqdm(range(s_code.shape[-2])):
            for j in range(s_code.shape[-1]):
                params = self.forward(
                    image[..., : i + 1, : j + 1], s_code[..., : i + 1, : j + 1]
                )
                weights, loc, scale = self.extract_params(
                    params[..., i : i + 1, j : j + 1]
                )
                image[..., i, j] = sample_mixture_model(weights, loc, scale)

        return image
