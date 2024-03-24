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

    default_kernel_size = (3, 3)

    def __init__(self,
                 channels,
                 kernel=None,
                 groups=1,
                 batchnorm=True,
                 block_type=None,
                 gated=None):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError(
                "kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        pad = [k // 2 for k in kernel]
        self.gated = gated

        modules = []
        if block_type == 'cabcab':
            for i in range(2):
                conv = nn.Conv2d(channels,
                                 channels,
                                 kernel[i],
                                 padding=pad[i],
                                 groups=groups)
                modules.append(conv)
                modules.append(nn.Mish())
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))

        elif block_type == 'bacbac':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nn.Mish())
                conv = nn.Conv2d(channels,
                                 channels,
                                 kernel[i],
                                 padding=pad[i],
                                 groups=groups)
                modules.append(conv)
        else:
            raise ValueError("Unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer2d(channels, 1))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x) + x


class GateLayer2d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
