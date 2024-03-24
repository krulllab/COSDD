from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nn import Conv


class SDecoder(nn.Module):
    """Simple network for predicting noisy image from the signal code.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        s_code_channels (int): Number of channels in the input signal code.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the network.
        kernel_size (int): Size of the convolutional kernel.
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
    """

    def __init__(
        self,
        colour_channels,
        s_code_channels,
        n_filters=64,
        n_layers=4,
        kernel_size=3,
        checkpointed=False,
    ):
        super().__init__()
        self.checkpointed = checkpointed
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
            )
        )

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

    @staticmethod
    def loss(x, y):
        return F.mse_loss(x, y, reduction="none")
