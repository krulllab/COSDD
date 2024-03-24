import random

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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
