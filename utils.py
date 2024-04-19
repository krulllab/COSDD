import random

import torch
import numpy as np
from sklearn.linear_model import LinearRegression


class RandomCrop3D:
    """Randomly crop a 3D tensor"""
    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, img):
        D, H, W = self.size
        d, h, w = img.size()[1:]
        if d < D or h < H or w < W:
            raise ValueError(f"Image size {img.size()} is smaller than crop size {self.size}")
        
        d_start = random.randint(0, d - D)
        h_start = random.randint(0, h - H)
        w_start = random.randint(0, w - W)
        return img[:, d_start:d_start + D, h_start:h_start + H, w_start:w_start + W]


def autocorrelation(arrs, max_lag=25):
    """ Compute the spatial autocorrelation of a list of arrays.
    Args:
        arrs: list of arrays
        max_lag: int, the maximum lag to compute the autocorrelation for
    Returns:
        result: 2D tensor, the autocorrelation of the arrays
    """
    if not isinstance(arrs, list):
        arrs = [arrs]
    covar = torch.zeros((max_lag, max_lag))
    covar_denom = torch.zeros((max_lag, max_lag))
    var = 0
    var_denom = 0
    for a in arrs:
        a = a - a.mean()
        for i in range(max_lag):
            for j in range(max_lag):
                c = (a[..., :a.shape[-2]-i, :a.shape[-1]-j] * a[..., i:, j:]).sum()
                n = a[..., :a.shape[-2]-i, :a.shape[-1]-j].numel()
                covar[i, j] += c
                covar_denom[i, j] += n
        var += (a**2).sum()
        var_denom += a.numel()
    covar = covar / covar_denom
    var = var / var_denom

    ac = covar / var
    return ac


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
