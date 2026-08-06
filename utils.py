import random
from pathlib import Path
from typing import Literal, Iterable
from typing import Sequence, Tuple
from joblib import Parallel, delayed

from torchvision.transforms.v2 import Transform
from pytorch_lightning import LightningDataModule
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from sklearn.linear_model import LinearRegression


def patchify(x_list, patch_size):
    """Patchify an n-dimensional array into non-overlapping patches.

    If patch_size is not a divisor of the array shape, the edge patches will overlap
    with the adjacent patches.
    Args:
        x (torch.Tensor): Array to patchify. Shape [C, Z | Y | X]
        patch_size (tuple): Size of the patches. Shape [z | y | x]
    Returns:
        patches (torch.Tensor): Patchified array. Shape [n_patches, C, z | y | x]
    """
    patched = []
    for x in x_list:
        dimensions = x.ndim - 1
        assert (
            len(patch_size) == dimensions
        ), "Patch size must have the same number of dimensions as the array"

        remainders = [s % p for s, p in zip(x.shape[1:], patch_size)]
        remainder_dims = [i for i, r in enumerate(remainders) if r > 0]
        for d in remainder_dims:
            pad = torch.narrow(x, d + 1, x.size(d + 1) - patch_size[d], patch_size[d])
            x = torch.narrow(x, d + 1, 0, x.size(d + 1) - remainders[d])
            x = torch.cat([x, pad], dim=d + 1)

        for i in range(dimensions):
            x = x.unfold(i + 1, patch_size[i], patch_size[i])
        x = torch.movedim(x, 0, dimensions)
        x = x.flatten(0, dimensions - 1)
        patched.extend(x)

    return patched


def unpatchify(x_list, original_shapes, patch_size):
    unpatched = []
    for shape in original_shapes:
        # Original shape details
        C, *original_dims = shape
        dimensions = len(original_dims)

        # Calculate the number of patches along each spatial dimension
        num_patches_per_dim = [
            orig_dim // p + ((orig_dim % p) > 0) * 1
            for orig_dim, p in zip(original_dims, patch_size)
        ]
        num_patches = np.prod(num_patches_per_dim)
        patches = x_list[:num_patches]
        if isinstance(patches, list):
            patches = torch.stack(patches, 0)
        x_list = x_list[num_patches:]
        # Reshape patches to the grid of patches
        patches = torch.unflatten(patches, 0, num_patches_per_dim)
        patches = torch.movedim(patches, dimensions, 0)
        for _ in reversed(range(dimensions)):
            patches = patches.movedim(-1, -dimensions)
            patches = patches.flatten(-dimensions - 1, -dimensions)

        remainders = [s % p for s, p in zip(original_dims, patch_size)]
        remainder_dims = [i for i, r in enumerate(remainders) if r > 0]
        for d in remainder_dims:
            pad = torch.narrow(
                patches, d + 1, patches.size(d + 1) - remainders[d], remainders[d]
            )
            patches = torch.narrow(patches, d + 1, 0, patches.size(d + 1) - patch_size[d])
            patches = torch.cat((patches, pad), dim=d + 1)
        unpatched.append(patches)
    return unpatched


def percentile(
    tensors: Iterable[torch.Tensor],
    q: float,
):
    tensors = [torch.flatten(x) for x in tensors]
    tensors = torch.cat(tensors)
    p = np.percentile(tensors, q)
    return p


def mean_std(tensors):
    count = 0
    mean = torch.tensor(0.0)
    M2 = torch.tensor(0.0)

    for tensor in tensors:
        tensor = torch.flatten(tensor).to(torch.float)
        batch_count = tensor.numel()
        batch_mean = torch.mean(tensor)
        batch_M2 = torch.var(tensor, unbiased=False) * batch_count

        delta = batch_mean - mean
        new_count = count + batch_count
        mean += delta * batch_count / new_count
        M2 += batch_M2 + delta ** 2 * count * batch_count / new_count
        count = new_count

    std = torch.sqrt(M2 / (count - 1))
    return mean, std


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


class RandomCrop(Transform):
    def __init__(
        self,
        size: Sequence[int],
        pad_if_needed: bool = True,  # TODO: Could hide size errors
        padding_value: float = 0.0,
    ):
        super().__init__()
        self.size = tuple(size)
        self.pad_if_needed = pad_if_needed
        self.padding_value = padding_value

    # -------------------------
    # helpers
    # -------------------------
    def _spatial_shape(self, x: torch.Tensor) -> Tuple[int, ...]:
        if x.ndim == len(self.size):
            return tuple(x.shape)
        elif x.ndim == len(self.size) + 1:
            return tuple(x.shape[1:])
        else:
            raise ValueError(
                f"Unsupported input shape {tuple(x.shape)} for crop size {self.size}"
            )

    def _pad_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial_shape(x)
        pad = []

        for current, target in zip(reversed(spatial), reversed(self.size)):
            if current < target:
                total = target - current
                before = total // 2
                after = total - before
            else:
                before = after = 0
            pad.extend([before, after])

        if any(pad):
            x = F.pad(x, pad, value=self.padding_value)

        return x

    # -------------------------
    # v2 API
    # -------------------------
    def make_params(self, flat_inputs):
        x = flat_inputs[0]

        if self.pad_if_needed:
            x = self._pad_if_needed(x)

        spatial = self._spatial_shape(x)

        offsets = []
        for current, target in zip(spatial, self.size):
            if current == target:
                offsets.append(0)
            else:
                offsets.append(random.randint(0, current - target))

        return {"offsets": tuple(offsets)}

    def transform(self, x, params):
        if not isinstance(x, torch.Tensor):
            return x

        if self.pad_if_needed:
            x = self._pad_if_needed(x)

        offsets = params["offsets"]

        slices = []
        if x.ndim == len(self.size) + 1:
            slices.append(slice(None))  # channel dim

        for offset, size in zip(offsets, self.size):
            slices.append(slice(offset, offset + size))

        return x[tuple(slices)]

    def extra_repr(self):
        return (
            f"size={self.size}, "
            f"pad_if_needed={self.pad_if_needed}, "
            f"padding_value={self.padding_value}"
        )


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
        return image.to(torch.float32)


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
        self.n_iters = np.prod(low_snr[0].shape[1:]) // np.prod(rand_crop_size)

    def setup(self, stage):
        rand_crop = RandomCrop(self.rand_crop_size, pad_if_needed=True)  # TODO: could hide size errors
        random.shuffle(self.low_snr)
        train_set = self.low_snr[: int(len(self.low_snr) * self.train_split)]
        val_set = self.low_snr[int(len(self.low_snr) * self.train_split) :]

        self.train_set = TrainDataset(
            train_set,
            n_iters=self.n_iters,
            transform=rand_crop,
        )
        self.val_set = TrainDataset(
            val_set,
            n_iters=self.n_iters,
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
        return image.to(torch.float32)


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
            "continue-checkpoint": None,
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
                "monte-carlo-kl": False,  # unused. For backwards compatibility
                "use-direct-denoiser": True,
                "direct-denoiser-loss": "MSE",
            },
            "hyper-parameters": {
                "s-code-channels": 64,
                "number-layers": 8,
                "scale-initialisation": False,
                "number-components": 3,
                "discretised": False,
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
            "use-direct-denoiser": True,
            "use-direct-denoiser": True,
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
    if not predict:
        if config_dict["train-parameters"]["crop-size"] is not None:
            if config_dict["data"]["number-dimensions"] != len(config_dict["train-parameters"]["crop-size"]):
                raise ValueError(
                        f'Random crop size: {config_dict["train-parameters"]["crop-size"]} has different dimensions to data: {config_dict["data"]["number-dimensions"]}'
                    )
    return config_dict


# def axes_to_SCZYX(images, axes, n_dimensions):
#     spatial_axes = [d for d in "TZYX" if d in axes]
#     spatial_axes = spatial_axes[-n_dimensions:]
#     sample_axes = [d for d in "STZYXC" if d in axes and d not in spatial_axes]
#     target_axes = sample_axes + spatial_axes
#     target_transpose = [axes.index(d) for d in target_axes]
#     missing_axes = [i for i in range(len(axes)) if i not in target_transpose]
#     target_transpose = missing_axes + target_transpose
#     images = [i.transpose(target_transpose) for i in images]
#     if "C" not in axes:
#         images = [np.expand_dims(i, -n_dimensions - 1) for i in images]
#     images = [
#         i.reshape(-1, i.shape[-n_dimensions - 1], *i.shape[-n_dimensions:])
#         for i in images
#     ]
#     return images
def axes_to_SCZYX(images, axes, n_dimensions):
    spatial_axes = [d for d in "TZYX" if d in axes]
    spatial_axes = spatial_axes[-n_dimensions:]
    sample_axes = [d for d in "STZYXC" if d in axes and d not in spatial_axes]
    target_axes = sample_axes + spatial_axes
    target_transpose = [axes.index(d) for d in target_axes]
    missing_axes = [i for i in range(len(axes)) if i not in target_transpose]
    target_transpose = missing_axes + target_transpose

    # Convert list of arrays to a single array if possible (saves RAM)
    try:
        images = np.asarray(images)  # Only if shapes match
    except ValueError:
        pass  # fallback if images are of different shapes

    if isinstance(images, np.ndarray):
        # Transpose once
        images = images.transpose([0] + [i + 1 for i in target_transpose])  # adjust for batch dim

        # Expand channels if not present
        if "C" not in axes:
            images = np.expand_dims(images, -n_dimensions - 1)

        # Reshape in one go
        new_shape = (
            -1,
            images.shape[-n_dimensions - 1],  # C
            *images.shape[-n_dimensions:],   # ZYX
        )
        result = images.reshape(new_shape)
        result = [torch.from_numpy(i) for i in result]
    else:
        # Fallback for irregular shapes
        result = []
        for i in images:
            transposed = i.transpose(target_transpose)
            if "C" not in axes:
                transposed = np.expand_dims(transposed, -n_dimensions - 1)
            reshaped = transposed.reshape(
                -1, transposed.shape[-n_dimensions - 1], *transposed.shape[-n_dimensions:]
            )
            result.extend(reshaped)
        result = [torch.from_numpy(i) for i in result]
    return result


def SCZYX_to_axes(images, original_axes, original_sizes, n_dimensions):
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
        images_list.append(images[: int(sample_counts[i])])
        images_list[-1] = np.stack(images_list[-1], 0)
        images = images[int(sample_counts[i]) :]
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
    elif file_type == ".tif" or file_type == ".tiff":
        from tifffile import imread

        imread_fn = imread
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
    return_file_names: bool = False,
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
    # paralleliser = Parallel(n_jobs=32)  # TODO: set for number of available cores
    # images_and_names = paralleliser(
    #     delayed(imread_fn)(f)
    #     for f in tqdm(files)
    # )
    images_and_names = [(imread_fn(f), f) for f in tqdm(files)]
    images = [i[0] for i in images_and_names]
    files = [i[1] for i in images_and_names]
    if return_file_names:
        file_names = []
        for file in files:
            for path in paths:
                if file.is_relative_to(path):
                    file_names.append(file.relative_to(path))
    # images = [imread_fn(f) for f in tqdm(files)]
    original_sizes = []
    for i in images:
        original_size = np.array(i.shape)
        original_sizes.append(original_size)
    images[0].ndim == len(axes) or _raise(
        ValueError(f"Axes {axes} do not match shape of images: {images[0].shape}")
    )
    images = axes_to_SCZYX(images, axes, n_dimensions)
    if not return_file_names:
        return images, original_sizes
    else:
        return images, original_sizes, file_names
