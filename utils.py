import random

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def patchify(x, patch_size):
    """Patchify an n-dimensional array into non-overlapping patches.

    If patch_size is not a divisor of the array shape, the edge patches will overlap
    with the adjacent patches.
    Args:
        x (torch.Tensor): Array to patchify. Shape [B, C, Z | Y | X]
        patch_size (tuple): Size of the patches. Shape [z | y | x]
    Returns:
        patches (torch.Tensor): Patchified array. Shape [B * n_patches, C, z | y | x]
    """
    dimensions = x.ndim - 2
    assert (
        len(patch_size) == dimensions
    ), "Patch size must have the same number of dimensions as the array"

    remainders = [s % p for s, p in zip(x.shape[2:], patch_size)]
    remainder_dims = [i for i, r in enumerate(remainders) if r > 0]
    for d in remainder_dims:
        pad = torch.narrow(x, d + 2, x.size(d + 2) - patch_size[d], patch_size[d]).contiguous()
        x = torch.narrow(x, d + 2, 0, x.size(d + 2) - remainders[d]).contiguous()
        x = torch.cat([x, pad], dim=d + 2)

    for i in range(dimensions):
        x = x.unfold(i + 2, patch_size[i], patch_size[i])
    first_dims = [0] + [i + 2 for i in range(dimensions)]
    remaining_dims = [1] + [i + dimensions + 2 for i in range(dimensions)]
    new_dims = first_dims + remaining_dims
    x = x.permute(new_dims).contiguous()
    x = x.flatten(0, dimensions)

    return x


def autocorrelation(arr, max_lag=25):
    """Compute the autocorrelation over the last 2 dimensions of an arrays.
    Args:
        arr (torch.Tensor): Array with shape (..., H, W)
        max_lag (int): The maximum lag to compute the autocorrelation over
    Returns:
        result (torch.Tensor): 2D array of autocorrelation values
    """
    covar = torch.zeros(max_lag, max_lag)

    arr = arr - arr.mean()
    for i in range(max_lag):
        for j in range(max_lag):
            c = (arr[..., :arr.shape[-2]-i, :arr.shape[-1]-j] * arr[..., i:, j:]).mean()
            covar[i, j] = c

    var = (arr**2).mean()
    ac = covar / var
    return ac


class RandomCrop:
    """
    Randomly crops n-dimensional tensor to given size.

    Infers input tensor dimensions from len(output_size).

    Args:
        output_size (tuple): Desired output size of the crop.
    """
    def __init__(self, output_size):
        self.output_size = output_size
        self.n_dims = len(output_size)

    def __call__(self, x):
        x_size = x.size()[1:]
        assert all(xs >= os for xs, os in zip(x_size, self.output_size))

        start_idxs = [random.randint(0, xs - os) for xs, os in zip(x_size, self.output_size)]
        end_idxs = [si + os for si, os in zip(start_idxs, self.output_size)]
        crop = [slice(0, x.size(0))] 
        crop += [slice(si, ei) for si, ei in zip(start_idxs, end_idxs)]

        return x[crop]


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images, n_iters=1, transform=None):
        self.images = images
        self.n_images = len(images)
        self.n_iters = n_iters
        self.transform = transform

    def __len__(self):
        return self.n_images * self.n_iters

    def __getitem__(self, idx):
        idx = idx % self.n_images
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

class PredictDataset(torch.utils.data.Dataset):

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


def minimise_mse(x, y):
    x_ = x.flatten().reshape(-1, 1)
    y_ = y.flatten().reshape(-1, 1)

    reg = LinearRegression().fit(x_, y_)
    a = reg.coef_
    b = reg.intercept_
    return a * x + b


def normalise(x):
    low = np.percentile(x, 0.1)
    high = np.percentile(x, 99.9)
    x = (x - low) / (high - low)
    return x
