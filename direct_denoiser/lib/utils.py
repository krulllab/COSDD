import torch
from torch import nn


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """
    Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, depth, height, width) to new depth, height, and width given by a tuple.

    Args:
        x (torch.Tensor): Input volume tensor.
        size (tuple): Desired size (depth, height, width).
        mode (str): Mode, either 'pad' or 'crop'.

    Returns:
        The padded or cropped tensor.
    """

    assert x.dim() == 5 and len(size) == 3
    size = tuple(size)
    x_size = x.size()[2:5]

    # Calculate the difference in each dimension
    d_diff, h_diff, w_diff = (size[0] - x_size[0], size[1] - x_size[1], size[2] - x_size[2])

    if mode == 'pad':
        # Ensure we're not trying to pad to a smaller size
        if d_diff < 0 or h_diff < 0 or w_diff < 0:
            raise ValueError('Cannot pad to a smaller size. Current size: {}, target size: {}'.format(x_size, size))
        # Calculate padding on each side
        padding = [w_diff // 2, w_diff - w_diff // 2, h_diff // 2, h_diff - h_diff // 2, d_diff // 2,
                   d_diff - d_diff // 2]
        return nn.functional.pad(x, padding, mode='constant', value=0)

    elif mode == 'crop':
        # Ensure we're not trying to crop to a larger size
        if d_diff > 0 or h_diff > 0 or w_diff > 0:
            raise ValueError('Cannot crop to a larger size. Current size: {}, target size: {}'.format(x_size, size))
        # Calculate starting and ending indices
        d_start, h_start, w_start = -d_diff // 2, -h_diff // 2, -w_diff // 2
        d_end, h_end, w_end = d_start + size[0], h_start + size[1], w_start + size[2]
        return x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]


def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')
