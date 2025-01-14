import random
from pathlib import Path
from typing import Literal

from pytorch_lightning import LightningDataModule
import torch
import numpy as np
from sklearn.linear_model import LinearRegression


def patchify(x, patch_size):
    """Patchify an n-dimensional array into non-overlapping patches.

    If patch_size is not a divisor of the array shape, the edge patches will overlap
    with the adjacent patches.
    Args:
        x (torch.Tensor): Array to patchify. Shape [S, C, Z | Y | X]
        patch_size (tuple): Size of the patches. Shape [z | y | x]
    Returns:
        patches (torch.Tensor): Patchified array. Shape [S * n_patches, C, z | y | x]
    """
    dimensions = x.ndim - 2
    assert (
        len(patch_size) == dimensions
    ), "Patch size must have the same number of dimensions as the array"

    remainders = [s % p for s, p in zip(x.shape[2:], patch_size)]
    remainder_dims = [i for i, r in enumerate(remainders) if r > 0]
    for d in remainder_dims:
        pad = torch.narrow(x, d + 2, x.size(d + 2) - patch_size[d], patch_size[d])
        x = torch.narrow(x, d + 2, 0, x.size(d + 2) - remainders[d])
        x = torch.cat([x, pad], dim=d + 2)

    for i in range(dimensions):
        x = x.unfold(i + 2, patch_size[i], patch_size[i])
    x = torch.movedim(x, 1, 1 + dimensions)
    x = x.flatten(0, dimensions)

    return x


def unpatchify(patches, original_shape):
    """
    Reverse the patchification process and return the patched tensor to its original shape.

    Args:
        patches (torch.Tensor): Patchified tensor. Shape [S * n_patches, C, z | y | x]
        original_shape (tuple): The original shape of the tensor before patchification [S, C, Z | Y | X]
        patch_size (tuple): Size of each patch [z | y | x]

    Returns:
        torch.Tensor: Reconstructed tensor with shape [S, C, Z | Y | X]
    """
    # Original shape details
    S, C, *original_dims = original_shape
    dimensions = len(original_dims)
    patch_size = patches.shape[-dimensions:]

    # Calculate the number of patches along each spatial dimension
    num_patches = [
        orig_dim // p + ((orig_dim % p) > 0) * 1
        for orig_dim, p in zip(original_dims, patch_size)
    ]
    # Reshape patches to the grid of patches
    patches = torch.unflatten(patches, 0, [S, *num_patches])
    # Move the channel axis back
    patches = patches.movedim(1 + dimensions, 1)
    for _ in reversed(range(dimensions)):
        patches = patches.movedim(-1, -dimensions)
        patches = patches.flatten(-dimensions - 1, -dimensions)

    remainders = [s % p for s, p in zip(original_dims, patch_size)]
    remainder_dims = [i for i, r in enumerate(remainders) if r > 0]
    for d in remainder_dims:
        pad = torch.narrow(
            patches, d + 2, patches.size(d + 2) - remainders[d], remainders[d]
        )
        patches = torch.narrow(patches, d + 2, 0, patches.size(d + 2) - patch_size[d])
        patches = torch.cat((patches, pad), dim=d + 2)
    return patches


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
            c = (
                arr[..., : arr.shape[-2] - i, : arr.shape[-1] - j] * arr[..., i:, j:]
            ).mean()
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

        start_idxs = [
            random.randint(0, xs - os) for xs, os in zip(x_size, self.output_size)
        ]
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


class DataModule(LightningDataModule):
    def __init__(
        self,
        low_snr,
        batch_size=4,
        rand_crop_size=(256, 256),
        train_split=0.9,
    ):
        super().__init__()
        self.low_snr = low_snr
        self.batch_size = batch_size
        self.rand_crop_size = rand_crop_size
        self.train_split = train_split

    def setup(self, stage):
        n_iters = np.prod(self.low_snr.shape[2:]) // np.prod(self.rand_crop_size)
        rand_crop = RandomCrop(self.rand_crop_size)
        random.shuffle(self.low_snr)
        train_set = self.low_snr[: int(len(self.low_snr) * self.train_split)]
        val_set = self.low_snr[int(len(self.low_snr) * self.train_split) :]

        self.train_set = TrainDataset(
            train_set,
            n_iters=n_iters,
            transform=rand_crop,
        )
        self.val_set = TrainDataset(
            val_set,
            n_iters=n_iters,
            transform=rand_crop,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return val_loader


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


def get_defaults(config_dict, predict=False):
    """Loads default configuration options."""
    if not predict:
        defaults = {
            "model-name": None,
            "data": {
                "paths": None,
                "patterns": None,
                "axes": None,
                "number-dimensions": 2,
                "patch-size": None,
                "clip-outliers": False,
            },
            "train-parameters": {
                "batch-size": 4,
                "number-grad-batches": 4,
                "crop-size": [256, 256],
                "training-split": 0.9,
                "max-epochs": 1000,
                "max-time": "00:06:00:00",
                "patience": 50,
                "monte-carlo-kl": False,
                "use-direct-denoiser": True,
                "direct-denoiser-loss": "MSE",
            },
            "hyper-parameters": {
                "s-code-channels": 64,
                "number-layers": 8,
                "number-gaussians": 3,
                "noise-direction": "x",
            },
            "memory": {
                "precision": "bf16-mixed",
                "checkpointed": True,
                "gpu": [0],
            },
        }
    else:
        defaults = {
            "model-name": None,
            "n-samples": 100,
            "data": {
                "paths": None,
                "save-path": None,
                "patterns": None,
                "axes": None,
                "number-dimensions": 2,
                "patch-size": None,
                "clip-outliers": False,
            },
            "predict-parameters": {
                "batch-size": 1,
            },
            "memory": {
                "precision": "bf16-mixed",
                "gpu": [0],
            },
        }
    for config in config_dict.keys():
        if config not in defaults.keys():
            raise ValueError(f"`{config}` is not a valid configuration option")
        elif isinstance(defaults[config], dict):
            for subconfig in config_dict[config].keys():
                if subconfig not in defaults[config].keys():
                    raise ValueError(
                        f"`{config}: {subconfig}` is not a valid configuration option"
                    )
    for config in defaults:
        if config not in config_dict.keys():
            config_dict[config] = defaults[config]
        elif isinstance(defaults[config], dict):
            for subconfig in defaults[config]:
                if subconfig not in config_dict[config].keys():
                    config_dict[config][subconfig] = defaults[config][subconfig]
    if config_dict["data"]["patch-size"] is not None and predict == False:
        for i, s in enumerate(config_dict["train-parameters"]["crop-size"]):
            if s > config_dict["data"]["patch-size"][i]:
                raise ValueError(
                    f'Random crop size: {config_dict["train-parameters"]["crop-size"]} is larger than patch size: {config_dict["data"]["patch-size"]}'
                )
    return config_dict


def axes_to_SCZYX(images, axes, n_dimensions):
    spatial_axes = [d for d in "TZYX" if d in axes]
    spatial_axes = spatial_axes[-n_dimensions:]
    sample_axes = [d for d in "STZYXC" if d in axes and d not in spatial_axes]
    target_axes = sample_axes + spatial_axes
    target_transpose = [axes.index(d) for d in target_axes]
    missing_axes = [i for i in range(len(axes)) if i not in target_transpose]
    target_transpose = missing_axes + target_transpose
    images = [i.transpose(target_transpose) for i in images]
    if "C" not in axes:
        images = [np.expand_dims(i, -n_dimensions - 1) for i in images]
    images = [
        i.reshape(-1, i.shape[-n_dimensions - 1], *i.shape[-n_dimensions:])
        for i in images
    ]
    return images


def SCZYX_to_axes(images, original_axes, original_sizes):
    n_dimensions = images.ndim - 2
    spatial_axes = [d for d in "TZYX" if d in original_axes]
    spatial_axes = spatial_axes[-n_dimensions:]
    sample_axes = [d for d in "STZYX" if d in original_axes and d not in spatial_axes]
    target_axes = sample_axes + spatial_axes
    missing_axes = [i for i in original_axes if i not in target_axes and i != "C"]
    sample_axes = missing_axes + sample_axes
    target_axes = sample_axes + spatial_axes
    sample_sizes = [
        [a[original_axes.index(s)] for s in sample_axes] for a in original_sizes
    ]
    sample_counts = [np.prod(s) for s in sample_sizes]
    images_list = []
    for i in range(len(original_sizes)):
        images_list.append(images[: sample_counts[i]])
        images = images[sample_counts[i] :]
    images_list = [
        image.reshape(*sample_sizes[i], image.shape[1], *image.shape[-n_dimensions:])
        for i, image in enumerate(images_list)
    ]
    if "C" not in original_axes:
        images_list = [image.squeeze(-n_dimensions - 1) for image in images_list]
    else:
        target_axes = target_axes[:-n_dimensions] + ["C"] + target_axes[-n_dimensions:]
    original_transpose = [target_axes.index(d) for d in original_axes]
    images_reshaped = [image.transpose(original_transpose) for image in images_list]
    return images_reshaped


def get_imread_fn(file_type):
    """Selects the function that will be used to load the data

    Edit this function to load file types that are not supported.
    The function should return a numpy array.
    """
    if file_type == ".czi":
        import czifile

        imread_fn = czifile.imread
    elif file_type == ".npy":
        imread_fn = np.load
    elif file_type == ".txt":
        imread_fn = np.loadtxt
    elif file_type == ".ARW":
        import rawpy

        def read_raw(f):
            with rawpy.imread(str(f)) as raw:
                rgb = raw.raw_image_visible.copy()
            return rgb

        imread_fn = read_raw
    else:
        from skimage import io

        imread_fn = io.imread
    return imread_fn


"""
BSD 3-Clause License

Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections import deque


def consume(iterator):
    deque(iterator, maxlen=0)


def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def axes_check_and_normalize(axes, n_dimensions):
    """
    S(ample), T(ime), C(hannel), Z, Y, X

    S(ample) can be repeated for extra dimensions that should be concatenated.
    E.g., for data is stored as shape (3, 2, 1, 128, 1024, 1024, 4),
    axes="SCSZYXS" returns images.shape=[3*1*4, 2, 128, 1024, 1024].
    """
    allowed = "STCZYX"
    axes is not None or _raise(ValueError("axis cannot be None."))
    axes = str(axes).upper()
    consume(
        a in allowed
        or _raise(
            ValueError("invalid axis '%s', must be one of %s." % (a, list(allowed)))
        )
        for a in axes
    )
    consume(
        axes.count(a) == 1
        or _raise(ValueError("axis '%s' cannot occur more than once." % a))
        for a in axes
        if a != "S"
    )
    matches_n_dimensions = sum([a in "TZYX" for a in axes]) >= n_dimensions
    if not matches_n_dimensions:
        raise ValueError(
            f"Images have fewer than {n_dimensions} spatial or temporal dimensions"
        )
    return axes


def load_data(
    paths: str | list,
    patterns: str | list = "*.tif",
    axes: str = "SYX",
    n_dimensions: Literal[1, 2, 3] = 2,
    dtype: torch.dtype = torch.float32,
):
    """Loads data from folders.

    Args:
        paths (str | list): Directories containing subdirectories with image files.
        patterns (str | list): Glob-style pattern to match images.
        axes (str): Semantics of axes of images as they are stored (assumed to be the same for all images).
        n_dimensions (int): Desired spatial dimensions of images once loaded (1, 2 or 3). Eg can be used to treat T(ime) as a spatial or sample dimension.
        dtype (np.dtype): Desired data type of loaded images.
    Returns:
        torch.tensor: The image data.
        list: The sizes of the images before conversion to pytorch [S, C, Z | Y | X].
    """
    if not isinstance(paths, list):
        paths = [paths]
    if not isinstance(patterns, list):
        patterns = [patterns]
    isinstance(n_dimensions, int) or _raise(
        TypeError(f"n_dimensions must be int but is {type(n_dimensions)}")
    )
    1 <= n_dimensions <= 3 or _raise(
        ValueError(f"n_dimensions must be 1, 2 or 3 but is {n_dimensions}")
    )
    axes = axes_check_and_normalize(axes, n_dimensions)
    files = []
    paths.sort()
    for path in paths:
        path = Path(path)
        path.exists() or _raise(FileNotFoundError(f'"{path}" does not exist'))
        if path.is_file():
            raise NotADirectoryError(
                f"{path} is a file. Use `patterns` to set file names."
            )
        for pattern in patterns:
            full_paths = list(path.glob(pattern))
            full_paths.sort()
            files.extend(full_paths)
    len(files) != 0 or _raise(FileNotFoundError("Could not find any images"))
    file_type = Path(files[0]).suffix
    imread_fn = get_imread_fn(file_type)
    images = [imread_fn(f) for f in files]
    original_sizes = []
    spatial_dims = [axes.index(i) for i in "XYZT"[:n_dimensions]]
    spatial_sizes = np.array(images[0].shape)[spatial_dims]
    for i in images:
        original_size = np.array(i.shape)
        i_spatial_size = original_size[spatial_dims]
        if np.all(i_spatial_size != spatial_sizes):
            _raise(
                ValueError(
                    f"Images do not all have the same spatial shape ({i.shape} and {images[0].shape})"
                )
            )
        original_sizes.append(original_size)
    images[0].ndim == len(axes) or _raise(
        ValueError(f"Axes {axes} do not match shape of images: {images[0].shape}")
    )
    images = axes_to_SCZYX(images, axes, n_dimensions)
    images = np.concatenate(images, 0).astype(float)
    return torch.from_numpy(images).to(dtype), original_sizes
