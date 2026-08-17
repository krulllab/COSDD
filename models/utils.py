import math
from numbers import Number

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily
import numpy as np
import matplotlib.pyplot as plt


class Rotate90(nn.Module):
    def __init__(self, k, dims):
        super().__init__()
        self.k = k
        self.dims = dims

    def forward(self, x, s_code):
        x = torch.rot90(x, k=self.k, dims=self.dims)
        s_code = torch.rot90(s_code, k=self.k, dims=self.dims)
        return x, s_code


def sample_mixture_model(logweights, loc, scale):
    px = MixtureSameFamily(Categorical(logits=logweights), Normal(loc, scale))
    return px.sample()


def interleave(tensor1, tensor2):
    """
    Interleaves two tensors along the channel dimension.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The interleaved tensor.

    Raises:
        AssertionError: If the shapes of tensor1 and tensor2 do not match.
    """
    assert tensor1.shape == tensor2.shape,  f"{tensor1.shape}, {tensor2.shape}"

    result = torch.stack((tensor1, tensor2), dim=2)

    N, C = result.shape[:2]
    result = result.reshape(N, 2 * C, *result.shape[3:])

    return result


def get_padded_size(size, n_downsc):
    """
    Calculates the necessary padded size of an image for a number of downscaling steps.

    Args:
        size (tuple): The desired size of the image as a tuple of (height, width).
        n_downsc (int): The number of downscaling steps.

    Returns:
        tuple: The padded size of the image as a tuple of (padded_height, padded_width).
    """
    dwnsc = [2 ** d for d in n_downsc]
    padded_size = [((s - 1) // d + 1) * d for s, d in zip(size, dwnsc)]

    return padded_size


def spatial_pad_crop(x, target_size):
    """
    Pads or crops the input tensor `x` to match the target size.

    Args:
        x (torch.Tensor): The input tensor to be padded or cropped.
        target_size (tuple): The target size to match.

    Returns:
        torch.Tensor: The padded or cropped tensor.
    """
    x_size = x.size()[2:]
    delta = [ts - xs for ts, xs in zip(target_size, x_size)]
    crop_delta = [(abs(d) // 2, abs(d) // 2 + abs(d) % 2) if d < 0 else (0, 0) for d in delta]
    pad_delta = [(d // 2, d // 2 + d % 2) if d > 0 else (0, 0) for d in delta]
    
    pad = []
    for d in reversed(pad_delta):
        pad.append(d[0])
        pad.append(d[1])
    x = nn.functional.pad(x, pad)
    x_size = x.size()[2:]
    crop = [slice(0, x.size(0)), slice(0,  x.size(1))]
    crop += [slice(d[0], xs - d[1]) for d, xs in zip(crop_delta, x_size)]
    return x[tuple(crop)]


def _to_4tuple(value, name):
    """Normalize an int/float-or-4-tuple argument into a 4-tuple."""
    if isinstance(value, Number):
        return (value, value, value, value)
    value = tuple(value)
    assert len(value) == 4, f"{name} must be a number or a 4-tuple, got {value!r}"
    return value


def _interp1d_along_dim(x, dim, out_size, align_corners):
    """
    Apply 1D linear interpolation along a single dimension `dim` of an
    arbitrary-rank tensor `x`, leaving all other dimensions untouched.

    This works by moving `dim` to the end, flattening every other dimension
    into a single "batch" axis, running ``F.interpolate(..., mode='linear')``
    (PyTorch's own audited 1D linear interpolation, so all of the
    align_corners / coordinate-mapping edge cases are handled exactly as
    PyTorch handles them), and then undoing the reshape/permute.
    """
    ndim = x.dim()
    if dim < 0:
        dim += ndim

    perm = [d for d in range(ndim) if d != dim] + [dim]
    x_perm = x.permute(perm).contiguous()
    lead_shape = x_perm.shape[:-1]
    in_size = x_perm.shape[-1]

    # F.interpolate(mode='linear') expects shape (N, C, W)
    x_flat = x_perm.reshape(1, -1, in_size)
    out_flat = F.interpolate(
        x_flat, size=out_size, mode="linear", align_corners=align_corners
    )
    out_perm = out_flat.reshape(*lead_shape, out_size)

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return out_perm.permute(inv_perm)


def quadlinear_interpolate(input, size=None, scale_factor=None, align_corners=False):
    """
    Quadrilinear ("quadlinear") interpolation of a tensor's four innermost
    spatial dimensions, extending ``torch.nn.functional.interpolate``'s
    ``linear`` / ``bilinear`` / ``trilinear`` modes (which handle 1, 2, and 3
    spatial dimensions respectively) to 4 spatial dimensions.

    Given an input of shape ``(b, c, t, d, h, w)``, each output value is a
    weighted average of the ``2**4 = 16`` nearest input corners, with weights
    equal to the product of the four per-axis linear-interpolation weights.
    This is mathematically identical to (and numerically matches) applying
    PyTorch's own 1D ``linear`` interpolation successively along each of the
    four axes, which is how this function is implemented -- interpolation
    weights are separable, so nesting the four 1D interpolations produces
    exactly the 16-corner weighted sum, without having to reimplement
    PyTorch's ``align_corners`` / coordinate-mapping logic by hand.

    Args:

        input (Tensor):

            Input tensor of shape ``(b, c, t, d, h, w)``.

        size (int or 4-tuple of int, optional):

            Output spatial size ``(t', d', h', w')``. Mutually exclusive
            with ``scale_factor``.

        scale_factor (float or 4-tuple of float, optional):

            Multiplier for the spatial size of each of the four dimensions.
            Mutually exclusive with ``size``. Output sizes are computed as
            ``floor(input_size * scale_factor)``, matching
            ``nn.functional.interpolate``'s default behavior.

        align_corners (bool, optional):

            Same meaning as in ``nn.functional.interpolate``: if ``True``,
            the corner pixels of input and output are aligned, preserving
            values at the corners. Default: ``False``.

    Returns:

        Tensor of shape ``(b, c, t', d', h', w')``.
    """
    assert input.dim() == 6, (
        "quadlinear_interpolate expects a 6D tensor of shape "
        f"(b, c, t, d, h, w), got shape {tuple(input.shape)}"
    )
    assert (size is None) != (scale_factor is None), (
        "Exactly one of `size` or `scale_factor` must be specified"
    )

    in_sizes = input.shape[2:]

    if size is not None:
        out_sizes = _to_4tuple(size, "size")
        out_sizes = tuple(int(s) for s in out_sizes)
    else:
        scale_factors = _to_4tuple(scale_factor, "scale_factor")
        out_sizes = tuple(
            int(math.floor(in_sizes[i] * scale_factors[i])) for i in range(4)
        )

    output = input
    # Interpolate each of the four spatial dims (2, 3, 4, 5) in turn. Order
    # doesn't matter mathematically since the operation is separable.
    for axis, out_size in zip((2, 3, 4, 5), out_sizes):
        if out_size == output.shape[axis]:
            continue
        output = _interp1d_along_dim(output, axis, out_size, align_corners)

    return output


class LinearUpsample(nn.Module):
    """
    Upsamples the input tensor `x` using linear/bilinear/trilinear interpolation.

    Args:
        x (torch.Tensor): The input tensor to be upsampled.
        scale_factor (int or tuple): The scale factor for the upsampling operation.

    Returns:
        torch.Tensor: The upsampled tensor.
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.modes = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}

    def forward(self, x):
        d = x.dim() - 2
        if d <= 3:
            mode = self.modes[d]
            return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=mode)
        elif d == 4:
            return quadlinear_interpolate(x, scale_factor=self.scale_factor)


class BatchNorm4d(nn.modules.batchnorm._BatchNorm):
    """
    Batch normalization for four spatial dimensions.

    Extends ``nn.BatchNorm1d`` / ``nn.BatchNorm2d`` / ``nn.BatchNorm3d`` to
    inputs with four spatial dimensions, i.e. tensors of shape ``(b, c, t,
    d, h, w)`` as produced by ``Conv4d`` and consumed by
    ``quadlinear_interpolate``. Per channel, statistics are computed over
    the batch dimension and all four spatial dimensions.

    ``nn.modules.batchnorm._BatchNorm`` already implements ``__init__`` and
    ``forward`` (via ``F.batch_norm``) in a way that doesn't depend on the
    number of spatial dimensions; only ``_check_input_dim`` differs between
    ``BatchNorm1d``/``2d``/``3d``, so that's all that needs overriding here.

    Args:
        num_features (int): Number of channels `c` in the input.
        eps (float, optional): Value added to the denominator for numerical
            stability. Default: 1e-5.
        momentum (float, optional): Value used for the running_mean and
            running_var computation, i.e. the exponential moving average
            factor. Can be set to `None` for a cumulative moving average
            instead. Default: 0.1.
        affine (bool, optional): If `True`, this module has learnable
            per-channel scale and shift parameters. Default: True.
        track_running_stats (bool, optional): If `True`, tracks the running
            mean and variance and uses them in eval mode. If `False`, always
            uses batch statistics, in both training and eval mode.
            Default: True.
    """

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError(
                f"expected 6D input (b, c, t, d, h, w), got {input.dim()}D input"
            )


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    canvas = figure.canvas
    width, height = canvas.get_width_height()
    canvas.draw()
    image = (
        np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        .reshape(height, width, 4)
        .transpose(2, 0, 1)
    )
    image = image / 255
    plt.close(figure)
    return image


class Conv4d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        dilation=1,
        groups=1,
        bias=True,
        device=None,
        dtype=None,
        bias_initializer=None,
        kernel_initializer=None,
    ):
        """
        Performs a 4D convolution of the ``(t, z, y, x)`` dimensions of a
        tensor with shape ``(b, c, t, d, h, w)`` with ``k`` filters. The output
        tensor will be of shape ``(b, k, t', d', h', w')``. ``(t', d', h',
        w')`` will be smaller than ``(t, d, h, w)`` if a padding smaller than
        half of the kernel size was chosen.

        Args:

            in_channels (int):

                Number of channels in the input image.

            out_channels (int):

                Number of channels produced by the convolution.

            kernel_size (int or tuple):

                Size of the convolving kernel.

            stride (int or tuple, optional):

                Stride of the convolution. Can be a single int (applied to all
                four dimensions) or a 4-tuple ``(stride_t, stride_d, stride_h,
                stride_w)``. Default: 1

            padding (int or tuple, optional):

                Zero- (or circular-) padding added to all four sides of the
                input. Can be a single int or a 4-tuple ``(padding_t,
                padding_d, padding_h, padding_w)``. Default: 0

            padding_mode (string, optional).

                Accepted values `zeros` and `circular`. Default: `zeros`

            dilation (int or tuple, optional):

                Spacing between kernel elements. Can be a single int or a
                4-tuple ``(dilation_t, dilation_d, dilation_h, dilation_w)``.
                Default: 1

            groups (int, optional):

                Number of blocked connections from input channels to output
                channels. ``in_channels`` and ``out_channels`` must both be
                divisible by ``groups``. Default: 1

            bias (bool, optional):

                If ``True``, adds a learnable bias to the output. Default:
                ``True``

            bias_initializer, kernel_initializer (callable):

                An optional initializer for the bias and the kernel weights.

        This operator realizes a 4D convolution by performing several 3D
        convolutions. The following example demonstrates how this works for a
        2D convolution as a sequence of 1D convolutions::

            I.shape == (h, w)
            k.shape == (U, V) and U%2 = V%2 = 1

            # we assume kernel is indexed as follows:
            u in [-U/2,...,U/2]
            v in [-V/2,...,V/2]

            (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                       = Σ_u (k[u]*I[i+u])[j]
            (k*I)[i]   = Σ_u k[u]*I[i+u]
            (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

            Example:

                I = [
                    [0,0,0],
                    [1,1,1],
                    [1,1,0],
                    [1,0,0],
                    [0,0,1]
                ]

                k = [
                    [1,1,1],
                    [1,2,1],
                    [1,1,3]
                ]

                # convolve every row in I with every row in k, comments show
                # output row the convolution contributes to
                (I*k[0]) = [
                    [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                    [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                    [2,2,1], # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                    [1,1,0], # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                    [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
                ]
                (I*k[1]) = [
                    [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                    [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                    [3,3,1], # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                    [2,1,0], # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                    [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
                ]
                (I*k[2]) = [
                    [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                    [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                    [4,2,1], # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                    [1,1,0], # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                    [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
                ]

                # the sum of all valid output rows gives k*I (here shown for
                # row 2)
                (k*I)[2] = (
                    [2,3,2] +
                    [3,3,1] +
                    [1,1,0] +
                ) = [6,7,3]
        """

        super(Conv4d, self).__init__()

        # ---------------------------------------------------------------------
        # Normalize constructor arguments to 4-tuples (t, d, h, w)
        # ---------------------------------------------------------------------
        kernel_size = _to_4tuple(kernel_size, "kernel_size")
        stride = _to_4tuple(stride, "stride")
        padding = _to_4tuple(padding, "padding")
        dilation = _to_4tuple(dilation, "dilation")

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if padding_mode not in ("zeros", "circular"):
            padding_mode = "zeros"
        # assert padding_mode in ("zeros", "circular"), (
        #     f"padding_mode must be 'zeros' or 'circular', got {padding_mode!r}"
        # )
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (t_k, d_k, h_k, w_k) = self.kernel_size
        (_, stride_d, stride_h, stride_w) = self.stride
        (_, padding_d, padding_h, padding_w) = self.padding
        (_, dilation_d, dilation_h, dilation_w) = self.dilation

        # Use a ModuleList to store layers to make the Conv4d layer trainable.
        # Every 3D conv shares the same groups value, so channel-wise grouping
        # is respected consistently for the extra (l) dimension as well: since
        # each per-frame Conv3d already restricts connectivity to within a
        # channel group, summing grouped conv3d outputs across l keeps the
        # groups independent from one another.
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(t_k):
            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(d_k, h_k, w_k),
                stride=(stride_d, stride_h, stride_w),
                padding=(padding_d, padding_h, padding_w),
                padding_mode=self.padding_mode,
                dilation=(dilation_d, dilation_h, dilation_w),
                groups=self.groups,
                bias=self.bias,
                device=device,
                dtype=dtype,
            )

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None and conv3d_layer.bias is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (b, c_i, t_i, d_i, h_i, w_i) = tuple(input.shape)
        (t_k, d_k, h_k, w_k) = self.kernel_size
        (stride_t, _, _, _) = self.stride
        (padding_t, _, _, _) = self.padding
        (dilation_t, _, _, _) = self.dilation

        # Compute the size of the output tensor along l using the standard
        # convolution output-size formula (identical to nn.Conv1d/2d/3d).
        t_o = (
            t_i + 2 * padding_t - dilation_t * (t_k - 1) - 1
        ) // stride_t + 1

        # Output tensors for each 3D frame
        frame_results = t_o * [None]

        # Convolve each kernel frame i with each input frame j. This mirrors
        # exactly how nn.Conv1d combines input positions with kernel taps:
        # for output position t_o and kernel tap i, the corresponding input
        # position (in unpadded coordinates) is
        #   j = t_o * stride_t - padding_t + i * dilation_t
        for i in range(t_k):
            conv3d_layer = self.conv3d_layers[i]

            for t_o in range(t_o):
                j = t_o * stride_t - padding_t + i * dilation_t

                if self.padding_mode == "circular":
                    # Wrap around: any j maps to a valid index modulo t_i
                    j = j % t_i
                else:
                    # Zero padding: out-of-range taps contribute nothing
                    if j < 0 or j >= t_i:
                        continue

                frame_conv3d = conv3d_layer(
                    input[:, :, j, :, :, :].view(b, c_i, d_i, h_i, w_i)
                )

                if frame_results[t_o] is None:
                    frame_results[t_o] = frame_conv3d
                else:
                    frame_results[t_o] = frame_results[t_o] + frame_conv3d

        return torch.stack(frame_results, dim=2)
