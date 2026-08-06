import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .nn import PixelCNNLayers as Layers
from .nn import Conv
from .utils import sample_mixture_model


def log_normal_pdf(x, loc, scale):
    a = -torch.log(scale)
    b = torch.log(torch.tensor(2, device=loc.device) * torch.pi)
    c = ((x - loc) / scale)**2
    return a - 0.5 * (b + c)


class DiscreteLogistic:
    def __init__(self, min_bound=0, max_bound=256):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.num_vals = max_bound - min_bound

    def __call__(self, y, means, log_scales, mixture_logits):
        inv_scales = torch.exp(-log_scales).to(y.dtype)

        y_range = self.max_bound - self.min_bound
        # explained in text
        epsilon = (0.5 * y_range) / (self.num_vals - 1)
        # convenience variable
        y = y.unsqueeze(-1)
        y = torch.repeat_interleave(y, means.shape[-1], -1)
        centered_y = y - means
        # inputs to our sigmoid functions
        upper_bound_in = inv_scales * (centered_y + epsilon)
        lower_bound_in = inv_scales * (centered_y - epsilon)
        # remember: cdf of logistic distr is sigmoid of above input format
        upper_cdf = torch.sigmoid(upper_bound_in)
        lower_cdf = torch.sigmoid(lower_bound_in)
        # finally, the probability mass and equivalent log prob
        prob_mass = upper_cdf - lower_cdf
        vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12)).to(y.dtype)

        # edges
        low_bound_log_prob = upper_bound_in - F.softplus(
            upper_bound_in
        ).to(y.dtype)  # log probability for edge case of 0 (before scaling)
        upp_bound_log_prob = -F.softplus(
            lower_bound_in
        ).to(y.dtype)  # log probability for edge case of 255 (before scaling)
        # middle
        mid_in = inv_scales * centered_y
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in).to(y.dtype)
        log_prob_mid = log_pdf_mid - torch.log((self.num_vals - 1) / 2).to(y.dtype)

        # Create a tensor with the same shape as 'y', filled with zeros
        log_probs = torch.zeros_like(centered_y)
        # conditions for filling in tensor
        is_near_min = y < self.min_bound + 1e-3
        is_near_max = y > self.max_bound - 1e-3
        is_prob_mass_sufficient = prob_mass > 1e-5 
        # And then fill it in accordingly
        # lower edge
        log_probs[is_near_min] = low_bound_log_prob[is_near_min]
        # upper edge
        log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
        # vanilla case
        log_probs[~is_near_min & ~is_near_max & is_prob_mass_sufficient] = vanilla_log_prob[
            ~is_near_min & ~is_near_max & is_prob_mass_sufficient
        ]
        # extreme case where prob mass is too small
        log_probs[~is_near_min & ~is_near_max & ~is_prob_mass_sufficient] = log_prob_mid[
            ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
        ]

        # modeling which mixture to sample from
        log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1, dtype=y.dtype)

        # log likelihood
        log_likelihood = torch.logsumexp(log_probs, dim=-1).to(y.dtype)

        return log_likelihood


class DiscreteLogistic:
    def __init__(self, min_bound=0, max_bound=256):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.num_vals = max_bound - min_bound

    def __call__(self, y, means, log_scales, mixture_logits):
        inv_scales = torch.exp(-log_scales).to(y.dtype)

        y_range = self.max_bound - self.min_bound
        # explained in text
        epsilon = (0.5 * y_range) / (self.num_vals - 1)
        # convenience variable
        y = y.unsqueeze(-1)
        y = torch.repeat_interleave(y, means.shape[-1], -1)
        centered_y = y - means
        # inputs to our sigmoid functions
        upper_bound_in = inv_scales * (centered_y + epsilon)
        lower_bound_in = inv_scales * (centered_y - epsilon)
        # remember: cdf of logistic distr is sigmoid of above input format
        upper_cdf = torch.sigmoid(upper_bound_in)
        lower_cdf = torch.sigmoid(lower_bound_in)
        # finally, the probability mass and equivalent log prob
        prob_mass = upper_cdf - lower_cdf
        vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12)).to(y.dtype)

        # edges
        low_bound_log_prob = upper_bound_in - F.softplus(
            upper_bound_in
        ).to(y.dtype)  # log probability for edge case of 0 (before scaling)
        upp_bound_log_prob = -F.softplus(
            lower_bound_in
        ).to(y.dtype)  # log probability for edge case of 255 (before scaling)
        # middle
        mid_in = inv_scales * centered_y
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in).to(y.dtype)
        log_prob_mid = log_pdf_mid - torch.log((self.num_vals - 1) / 2).to(y.dtype)

        # Create a tensor with the same shape as 'y', filled with zeros
        log_probs = torch.zeros_like(centered_y)
        # conditions for filling in tensor
        is_near_min = y < self.min_bound + 1e-3
        is_near_max = y > self.max_bound - 1e-3
        is_prob_mass_sufficient = prob_mass > 1e-5 
        # And then fill it in accordingly
        # lower edge
        log_probs[is_near_min] = low_bound_log_prob[is_near_min]
        # upper edge
        log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
        # vanilla case
        log_probs[~is_near_min & ~is_near_max & is_prob_mass_sufficient] = vanilla_log_prob[
            ~is_near_min & ~is_near_max & is_prob_mass_sufficient
        ]
        # extreme case where prob mass is too small
        log_probs[~is_near_min & ~is_near_max & ~is_prob_mass_sufficient] = log_prob_mid[
            ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
        ]

        # modeling which mixture to sample from
        log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1, dtype=y.dtype)

        # log likelihood
        log_likelihood = torch.logsumexp(log_probs, dim=-1).to(y.dtype)

        return log_likelihood


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
        n_components=5,
        discretised=False,
        data_min=0,
        data_max=256,
        gated=False,
        checkpointed=False,
        dimensions=2,
    ):
        super().__init__()
        noise_direction = noise_direction.lower()
        assert noise_direction in ("x", "y", "z", "none")
        self.n_components = n_components
        self.n_components = n_components
        self.colour_channels = colour_channels
        self.noise_direction = noise_direction
        self.discretised = discretised
        if self.discretised:
            self.discrete_logistic_pmf = DiscreteLogistic(min_bound=data_min, max_bound=data_max)

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

        c_out = n_components * colour_channels * 3
        c_out = n_components * colour_channels * 3
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
        logweights = params[:, 0::3].unfold(1, self.n_components, self.n_components)
        loc = params[:, 1::3].unfold(1, self.n_components, self.n_components)
        log_scale = params[:, 2::3].unfold(1, self.n_components, self.n_components)
        return logweights, loc, log_scale
        logweights = params[:, 0::3].unfold(1, self.n_components, self.n_components)
        loc = params[:, 1::3].unfold(1, self.n_components, self.n_components)
        log_scale = params[:, 2::3].unfold(1, self.n_components, self.n_components)
        return logweights, loc, log_scale
    
    def loglikelihood(self, x, params):
        logweights, loc, log_scale = self.extract_params(params)
        if self.discretised:
            log_p = self.discrete_logistic_pmf(x, loc, log_scale, logweights)
        else:
            scale = nn.functional.softplus(log_scale)
            logweights = nn.functional.log_softmax(logweights, dim=-1)
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
                logweights, loc, log_scale = self.extract_params(params[..., i : i + 1])
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "y":
            for i in tqdm(range(s_code.shape[-2]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1, :], s_code[..., : i + 1, :])
                logweights, loc, log_scale = self.extract_params(params[..., i : i + 1, :])
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1, :] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "z":
            for i in tqdm(range(s_code.shape[-3]), bar_format="{l_bar}{bar}|"):
                params = self(image[..., : i + 1, :, :], s_code[..., : i + 1, :, :])
                logweights, loc, log_scale = self.extract_params(
                    params[..., i : i + 1, :, :]
                )
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1, :, :] = sample_mixture_model(
                    logweights, loc, scale
                )

        return image
