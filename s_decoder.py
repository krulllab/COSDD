from torch import nn
import torch.nn.functional as F


class SDecoder(nn.Module):
    """ Simple network for predicting noisy image from the signal code.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        s_code_channels (int): Number of channels in the input signal code.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the network.
        kernel_size (int): Size of the convolutional kernel.
    """

    def __init__(
        self, colour_channels, s_code_channels, n_filters=64, n_layers=4, kernel_size=3
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be greater than 2")

        modules = [
            nn.Conv3d(
                s_code_channels,
                n_filters,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 2):
            modules.extend(
                [
                    nn.Conv3d(
                        n_filters,
                        n_filters,
                        kernel_size,
                        padding=kernel_size // 2,
                        padding_mode="reflect",
                    ),
                    nn.ReLU(),
                ]
            )
        modules.append(
            nn.Conv3d(
                n_filters,
                colour_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
            )
        )

        self.net = nn.Sequential(*modules)

    def forward(self, s_code):
        return self.net(s_code)

    @staticmethod
    def loss(x, y):
        return F.mse_loss(x, y, reduction="none")
