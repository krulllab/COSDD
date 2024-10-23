import yaml
import os
import argparse
import pickle
import math
import random
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

import utils
from models.lvae import LadderVAE
from models.pixelcnn import PixelCNN
from models.s_decoder import SDecoder
from models.unet import UNet
from models.hub import Hub


parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

assert torch.cuda.is_available()

with open(args.config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = utils.get_defaults(cfg)

low_snr = utils.load_data(
    cfg["data"]["paths"],
    cfg["data"]["patterns"],
    cfg["data"]["axes"],
    cfg["data"]["number-dimensions"],
)
if cfg["data"]["patch-size"] is not None:
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])

val_split = 1 - cfg["train-parameters"]["training-split"]
if math.floor((val_split) * len(low_snr)) == 0:
    print(
        f"Data of shape: {low_snr.size()} cannot be split {cfg["train-parameters"]["training-split"]}/\
          {round(val_split, 2)} train/validation along sample axis."
    )
    print("Automatically patching data...")
    val_patch_size = [
        math.ceil(
            low_snr.shape[-i] * (val_split ** (1 / cfg["data"]["number-dimensions"]))
        )
        for i in reversed(range(1, cfg["data"]["number-dimensions"] + 1))
    ]
    low_snr = utils.patchify(low_snr, patch_size=val_patch_size)
print(f"Noisy data shape: {low_snr.size()}")

if cfg["data"]["clip-outliers"]:
    clip_min = np.percentile(low_snr, 1)
    clip_max = np.percentile(low_snr, 99)
    clip_min_max = (clip_min, clip_max)
else:
    clip_min_max = None

print(
    f"Effective batch size: {cfg["train-parameters"]["batch-size"] * cfg["train-parameters"]["number-grad-batches"]}"
)
n_iters = math.prod(low_snr.shape[-cfg["data"]["number-dimensions"] :]) // math.prod(
    cfg["train-parameters"]["crop-size"]
)
transform = utils.RandomCrop(cfg["train-parameters"]["crop-size"])

idxs = list(range(len(low_snr)))
random.shuffle(idxs)
low_snr = low_snr[idxs]
train_set = low_snr[: int(len(low_snr) * cfg["train-parameters"]["training-split"])]
val_set = low_snr[int(len(low_snr) * cfg["train-parameters"]["training-split"]) :]

train_set = utils.TrainDataset(train_set, n_iters=n_iters, transform=transform)
val_set = utils.TrainDataset(val_set, n_iters=n_iters, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=cfg["train-parameters"]["batch-size"],
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=cfg["train-parameters"]["batch-size"],
    shuffle=False,
)

z_dims = [cfg["hyper-parameters"]["s-code-channels"] // 2] * cfg["hyper-parameters"][
    "number-layers"
]
max_size = max(cfg["train-parameters"]["crop-size"])
num_halves = math.floor(math.log2(max_size)) - 1
downsampling = [1] * cfg["hyper-parameters"]["number-layers"]
difference = max(cfg["hyper-parameters"]["number-layers"] - num_halves, 0)
i = 0
while difference > 0:
    for j in range(cfg["hyper-parameters"]["number-layers"] // 2):
        downsampling[i + j * 2] = 0
        difference -= 1
        if difference == 0:
            break
    i += 1

lvae = LadderVAE(
    colour_channels=low_snr.shape[1],
    img_size=cfg["train-parameters"]["crop-size"],
    s_code_channels=cfg["hyper-parameters"]["s-code-channels"],
    n_filters=cfg["hyper-parameters"]["s-code-channels"],
    z_dims=z_dims,
    downsampling=downsampling,
    monte_carlo_kl=cfg["train-parameters"]["monte-carlo-kl"],
    dimensions=cfg["data"]["number-dimensions"],
)

ar_decoder = PixelCNN(
    colour_channels=low_snr.shape[1],
    s_code_channels=cfg["hyper-parameters"]["s-code-channels"],
    kernel_size=5,
    noise_direction=cfg["hyper-parameters"]["noise-direction"],
    n_filters=64,
    n_layers=4,
    n_gaussians=cfg["hyper-parameters"]["number-gaussians"],
    dimensions=cfg["data"]["number-dimensions"],
)

s_decoder = SDecoder(
    colour_channels=low_snr.shape[1],
    s_code_channels=cfg["hyper-parameters"]["s-code-channels"],
    n_filters=cfg["hyper-parameters"]["s-code-channels"],
    dimensions=cfg["data"]["number-dimensions"],
)

if cfg["train-parameters"]["use-direct-denoiser"]:
    direct_denoiser = UNet(
        colour_channels=low_snr.shape[1],
        n_filters=cfg["hyper-parameters"]["s-code-channels"],
        n_layers=cfg["hyper-parameters"]["number-layers"],
        downsampling=downsampling,
        loss_fn=cfg["train-parameters"]["direct-denoiser-loss"],
        dimensions=cfg["data"]["number-dimensions"],
    )
else:
    direct_denoiser = None

mean_std_dims = [0, 2, 3] if cfg["data"]["number-dimensions"] == 2 else [0, 2, 3, 4]
data_mean = low_snr.mean(mean_std_dims, keepdims=True)
data_std = low_snr.std(mean_std_dims, keepdims=True)
hub = Hub(
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
    data_mean=data_mean,
    data_std=data_std,
    n_grad_batches=cfg["train-parameters"]["number-grad-batches"],
    checkpointed=cfg["memory"]["checkpointed"],
    clip_min_max=clip_min_max,
)

checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
logger = TensorBoardLogger(checkpoint_path)

if isinstance(cfg["memory"]["gpu"], int):
    cfg["memory"]["gpu"] = [cfg["memory"]["gpu"]]
trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=cfg["memory"]["gpu"],
    max_epochs=cfg["train-parameters"]["max-epochs"],
    max_time=cfg["train-parameters"]["max-time"],
    log_every_n_steps=len(train_set) // cfg["train-parameters"]["batch-size"],
    callbacks=[
        EarlyStopping(patience=cfg["train-parameters"]["patience"], monitor="val/elbo")
    ],
    precision=cfg["memory"]["precision"],
)

trainer.fit(hub, train_loader, val_loader)
current_time = time.strftime('%d-%m-%y-%X', time.localtime())
trainer.save_checkpoint(os.path.join(checkpoint_path, f"final_model-{current_time}.ckpt"))
with open(os.path.join(checkpoint_path, f"training-config-{current_time}.pkl"), "wb") as f:
    pickle.dump(cfg, f)
