import torch
from torch import nn
from tqdm import tqdm

from .nn import PixelCNNLayers as Layers
from .nn import Conv
from .utils import sample_mixture_model


def log_normal_pdf(x, loc, scale):
    a = -torch.log(scale)
    b = torch.log(torch.tensor(2, device=loc.device) * torch.pi)
    c = ((x - loc) / scale)**2
    return a - 0.5 * (b + c)


class PixelCNN(nn.Module):
    """Autoregressive decoder.

    Causal convolutions are one dimensional, with shape (1, kernel_size).
    To implement veritcal kernels, input is rotated 90 degrees on entry and
    rotated back on exit.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        s_code_channels (int): Number of channels in the decoded signal code.
        kernel_size (int): Size of the kernel in the convolutional layers.
        noise_direction (str): Axis along which receptive field runs.
        n_filters (int): Number of filters in the convolutional layers. Choose 'x', 'y', 'z' or 'none'
        n_layers (int): Number of layers.
        n_gaussians (int): Number of gaussians in the predictive mixture model.
        gated (int): Whether to use gated activations (A. Oord 2016).
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3)

    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        kernel_size,
        noise_direction="x",
        n_filters=64,
        n_layers=4,
        n_gaussians=5,
        gated=False,
        checkpointed=False,
        dimensions=2,
    ):
        super().__init__()
        noise_direction = noise_direction.lower()
        assert noise_direction in ("x", "y", "z", "none")
        self.n_gaussians = n_gaussians
        self.colour_channels = colour_channels
        self.noise_direction = noise_direction

        if noise_direction == 'none':
            kernel_size = 1

        # Uses grouped convolutions to ensure that each colour channel is
        # processed separately.
        if n_filters % colour_channels != 0:
            n_filters += colour_channels - (n_filters % colour_channels)

        self.layers = Layers(
            colour_channels=colour_channels,
            s_code_channels=s_code_channels,
            kernel_size=kernel_size,
            n_filters=n_filters,
            n_layers=n_layers,
            direction=noise_direction,
            gated=gated,
            checkpointed=checkpointed,
            dimensions=dimensions,
        )

        c_out = n_gaussians * colour_channels * 3
        self.out_conv = Conv(
            in_channels=n_filters,
            out_channels=c_out,
            kernel_size=1,
            groups=colour_channels,
            dimensions=dimensions,
        )

    def forward(self, x, s_code):
        out = self.layers(x, s_code)

        return self.out_conv(out)

    def extract_params(self, params):
        logweights = params[:, 0::3].unfold(1, self.n_gaussians, self.n_gaussians)
        logweights = nn.functional.log_softmax(logweights, dim=-1)
        loc = params[:, 1::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = params[:, 2::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = nn.functional.softplus(scale)
        return logweights, loc, scale
    
    def loglikelihood(self, x, params):
        logweights, loc, scale = self.extract_params(params)
        log_p_per_component = log_normal_pdf(x[..., None], loc, scale)
        log_weighted_p_per_component = logweights + log_p_per_component
        log_p = torch.logsumexp(log_weighted_p_per_component, dim=-1)

        return log_p

    @torch.no_grad()
    def sample(self, s_code):
        image = torch.zeros(s_code.shape[0], self.colour_channels, *s_code.shape[2:])
        image = image.to(s_code.get_device())

        if self.noise_direction == "x":
            for i in tqdm(range(s_code.shape[-1]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1], s_code[..., : i + 1])
                logweights, loc, scale = self.extract_params(params[..., i : i + 1])
                image[..., i : i + 1] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "y":
            for i in tqdm(range(s_code.shape[-2]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1, :], s_code[..., : i + 1, :])
                logweights, loc, scale = self.extract_params(params[..., i : i + 1, :])
                image[..., i : i + 1, :] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "z":
            for i in tqdm(range(s_code.shape[-3]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1, :, :], s_code[..., : i + 1, :, :])
                logweights, loc, scale = self.extract_params(
                    params[..., i : i + 1, :, :]
                )
                image[..., i : i + 1, :, :] = sample_mixture_model(
                    logweights, loc, scale
                )

        return image
