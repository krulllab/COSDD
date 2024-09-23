from torch import nn
import torch.nn.functional as F


class SDecoder(nn.Module):

    def __init__(self,
                 colour_channels,
                 s_code_channels,
                 n_filters=64,
                 n_layers=4,
                 kernel_size=3):
        super().__init__()
        if n_layers < 2:
            raise ValueError('n_layers must be greater than 2')

        modules = [
            nn.Conv2d(s_code_channels,
                      n_filters,
                      kernel_size,
                      padding=kernel_size // 2,
                      padding_mode='reflect'),
            nn.ReLU()
        ] + [
            nn.Conv2d(n_filters,
                      n_filters,
                      kernel_size,
                      padding=kernel_size // 2,
                      padding_mode="reflect"),
            nn.ReLU()
        ] * (n_layers - 2) + [
            nn.Conv2d(n_filters,
                      colour_channels,
                      kernel_size,
                      padding=kernel_size // 2,
                      padding_mode="reflect")
        ]

        self.net = nn.Sequential(*modules)

    def forward(self, s_code):
        return self.net(s_code)

    @staticmethod
    def loss(x, y):
        return F.mse_loss(x, y, reduction="none")
