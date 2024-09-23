import torch
from torch import nn

from .lib import Rotate90


class Shifted1DConvolution(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 first=False):
        super().__init__()
        self.first = first

        shift = dilation * (kernel_size - 1)

        self.pad = nn.ConstantPad2d((shift, 0, 0, 0), 0)

        self.conv = nn.Conv2d(in_channels,
                              out_channels, (1, kernel_size),
                              dilation=dilation)

        if self.first:
            kernel_mask = torch.ones((1, 1, 1, kernel_size))
            kernel_mask[..., -1] = 0
            self.register_buffer("kernel_mask", kernel_mask)

    def forward(self, x):
        x = self.pad(x)
        if self.first:
            self.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        s_code_channels,
        kernel_size,
        dilation=1,
        first=False,
    ):
        super().__init__()

        self.in_conv = Shifted1DConvolution(in_channels, out_channels,
                                            kernel_size, dilation, first)
        self.s_conv = nn.Conv2d(s_code_channels, out_channels, 1)
        self.act_fn = nn.ReLU()
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

        if out_channels == in_channels:
            self.do_skip = True
        else:
            self.do_skip = False

    def forward(self, x, s_code):
        feat = self.in_conv(x) + self.s_conv(s_code)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out, s_code


class Layers(nn.Module):

    def __init__(self, colour_channels, s_code_channels, kernel_size, n_filters,
                 n_layers, rotate90):
        super().__init__()

        middle_layer = n_layers // 2 + n_layers % 2

        layers = []
        if rotate90:
            layers.append(Rotate90(k=1))

        for i in range(n_layers):
            c_in = colour_channels if i == 0 else n_filters
            first = True if i == 0 else False
            dilation = 2**i if i < middle_layer else 2**(n_layers - i - 1)

            layers.append(
                Block(in_channels=c_in,
                      out_channels=n_filters,
                      s_code_channels=s_code_channels,
                      kernel_size=kernel_size,
                      dilation=dilation,
                      first=first))
            layers.append(
                Block(in_channels=n_filters,
                      out_channels=n_filters,
                      s_code_channels=s_code_channels,
                      kernel_size=kernel_size))

        if rotate90:
            layers.append(Rotate90(3))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, s_code):
        for layer in self.layers:
            x, s_code = layer(x, s_code)
        return x


class OutConvs(nn.Module):

    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()

        convs = []
        for i in range(n_layers):
            c_out = in_channels if i < (n_layers - 1) else out_channels
            convs.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=c_out,
                          kernel_size=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)
