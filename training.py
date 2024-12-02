import yaml
import os
import argparse
import math
import random
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

import utils
from models.get_models import get_models
from models.hub import Hub


parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

assert torch.cuda.is_available()

with open(args.config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = utils.get_defaults(cfg)

print("Loading data...")
low_snr = utils.load_data(
    cfg["data"]["paths"],
    cfg["data"]["patterns"],
    cfg["data"]["axes"],
    cfg["data"]["number-dimensions"],
)
if cfg["data"]["patch-size"] is not None:
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])

if math.ceil(cfg["train-parameters"]["training-split"] * len(low_snr)) == len(low_snr):
    val_split = round(1 - cfg["train-parameters"]["training-split"], 3)
    print(
        f'Data of shape: {low_snr.size()} cannot be split {cfg["train-parameters"]["training-split"]}/\
          {val_split} train/validation along sample axis.'
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
    print("Clippping min...")
    clip_min = np.percentile(low_snr, 1)
    print("Clippping max...")
    clip_max = np.percentile(low_snr, 99)
    low_snr = torch.clamp(low_snr, clip_min, clip_max)

print(
    f'Effective batch size: {cfg["train-parameters"]["batch-size"] * cfg["train-parameters"]["number-grad-batches"]}'
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

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(cfg, low_snr.shape[1])

# Each channel is normalised individually
mean_std_dims = [0, 2] + [i + 2 for i in range(1, cfg["data"]["number-dimensions"])]
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
        EarlyStopping(patience=cfg["train-parameters"]["patience"], monitor="elbo/val")
    ],
    plugins=[LightningEnvironment()],
    precision=cfg["memory"]["precision"],
)

trainer.fit(hub, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(checkpoint_path, f"final_model.ckpt"))
with open(os.path.join(checkpoint_path, 'training-config.yaml'), 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
