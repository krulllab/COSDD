import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nn import Conv
from .probability_density_functions import DiscreteLogistic


def softplus(x, beta):
    return 1/beta * torch.log(1 + torch.exp(beta * x))


class SDecoder(nn.Module):
    """Simple network for predicting noisy image from the signal code.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        s_code_channels (int): Number of channels in the input signal code.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the network.
        kernel_size (int): Size of the convolutional kernel.
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        n_filters=64,
        n_layers=4,
        kernel_size=3,
        discretised=False,
        data_min=0,
        data_max=255,
        num_vals=256,
        checkpointed=False,
        dimensions=2,
    ):
        super().__init__()
        self.checkpointed = checkpointed
        self.discretised = discretised
        if self.discretised:
            self.discrete_logistic_pmf = DiscreteLogistic(min_bound=data_min, max_bound=data_max, num_vals=num_vals)
        if n_layers < 2:
            raise ValueError("n_layers must be greater than 2")

        self.net = nn.ModuleList()
        self.net.append(
            nn.Sequential(
                Conv(
                    s_code_channels,
                    n_filters,
                    kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    dimensions=dimensions,
                ),
                nn.ReLU(),
            )
        )
        for _ in range(n_layers - 2):
            self.net.append(
                nn.Sequential(
                    Conv(
                        n_filters,
                        n_filters,
                        kernel_size,
                        padding=kernel_size // 2,
                        padding_mode="reflect",
                        dimensions=dimensions,
                    ),
                    nn.ReLU(),
                )
            )
        self.net.append(
            Conv(
                n_filters,
                colour_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
                dimensions=dimensions,
            )
        )
        if self.discretised:
            a = torch.tensor(0.1, requires_grad=True)
            b = torch.tensor(10.0, requires_grad=True)
            beta = torch.tensor(1.0, requires_grad=True)
            a = torch.nn.Parameter(a, requires_grad=True)
            b = torch.nn.Parameter(b, requires_grad=True)
            beta = torch.nn.Parameter(beta, requires_grad=True)
            self.register_parameter("a", a)
            self.register_parameter("b", b)
            self.register_parameter("beta", beta)

    def forward(self, s_code):
        for i, layer in enumerate(self.net):
            if i % 2 == 0 and self.checkpointed:
                s_code = checkpoint(
                    layer,
                    s_code,
                    use_reentrant=False,
                )
            else:
                s_code = layer(s_code)
        return s_code

    def loss(self, x, s_hat):
        if self.discretised:
            var = softplus(s_hat.detach() + self.b, beta=self.beta) * self.a
            log_scale = 0.5 * torch.log(var)
            # Add mixture dimension even though we're only using one component
            s_hat = s_hat[..., None]
            log_scale = log_scale[..., None]
            mixture_logits = torch.ones_like(s_hat)
            loss = -self.discrete_logistic_pmf(x, s_hat, log_scale, mixture_logits)
        else:
            loss = F.mse_loss(x, s_hat, reduction="none")

        return loss
