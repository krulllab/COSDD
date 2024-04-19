import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
    For example, bacbac has 2x (batchnorm, activation, conv).
    """

    default_kernel_size = (3, 3, 3)

    def __init__(self,
                 channels,
                 kernel_size=None,
                 groups=1,
                 block_type=None,
                 gated=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = self.default_kernel_size
        elif isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) != 3:
            raise ValueError(
                "kernel has to be None, int, or an iterable of length 3")
        assert all([k % 2 == 1 for k in kernel_size]), "kernel sizes have to be odd"
        kernel_size = list(kernel_size)
        if block_type is None:
            block_type = "bacbac"
        self.gated = gated

        component_dict = {
            "a": lambda: nn.Mish(),
            "b": lambda: nn.BatchNorm3d(channels),
            "c": lambda: nn.Conv3d(
                channels,
                channels,
                kernel_size,
                groups=groups,
                padding="same",
                padding_mode="replicate",
            ),
        }

        modules = []
        for component in block_type:
            modules.append(
                component_dict.get(component, f"Unrecognized component '{component}'")()
            )

        if gated:
            modules.append(GateLayer3d(channels, 1))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x) + x


class GateLayer3d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv3d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
