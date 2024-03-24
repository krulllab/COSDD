import torch
from torch import nn
from torch.distributions import Categorical, Normal, MixtureSameFamily


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
    dwnsc = 2 ** n_downsc
    padded_size = [((s - 1) // dwnsc + 1) * dwnsc for s in size]

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
    return x[crop]


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
        mode = self.modes[d]
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=mode)
