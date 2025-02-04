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


assert torch.cuda.is_available()

# Load configuration options
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

with open(args.config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# Sets configuration options not given to defaults and checks given arguments
cfg = utils.get_defaults(cfg)

print("Loading data...")
low_snr, _ = utils.load_data(
    cfg["data"]["paths"],
    cfg["data"]["patterns"],
    cfg["data"]["axes"],
    cfg["data"]["number-dimensions"],
)
if cfg["data"]["patch-size"] is not None:
    # Split data into non-overlapping patches
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])

# The loaded data is split into training and validation sets along the same axis, i.e. different images/volumes
# are used for training validation.
# If there are too few images for the chosen training/validation split (e.g. 0.9/0.1), individual images will
# have to be broken up into patches, and the patches randomly split into training/validation sets.
# This will try to do so automatically, but should be done manually by setting data: patch-size configuration option.
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
    # To avoid outliers causing problems, clip data values outside of 1st and 99th percentiles
    print("Clippping min...")
    clip_min = np.percentile(low_snr, 1)
    print("Clippping max...")
    clip_max = np.percentile(low_snr, 99)
    low_snr = torch.clamp(low_snr, clip_min, clip_max)

print(
    f'Effective batch size: {cfg["train-parameters"]["batch-size"] * cfg["train-parameters"]["number-grad-batches"]}'
)

datamodule = utils.DataModule(
    low_snr=low_snr,
    batch_size=cfg["train-parameters"]["batch-size"],
    rand_crop_size=cfg["train-parameters"]["crop-size"],
    train_split=cfg["train-parameters"]["training-split"],
)
# Load models
lvae, ar_decoder, s_decoder, direct_denoiser = get_models(cfg, low_snr.shape[1])

# Each channel is normalised individually.
mean_std_dims = [0, 2] + [i + 2 for i in range(1, cfg["data"]["number-dimensions"])]
if "64" in str(cfg["memory"]["precision"]):
    dtype = torch.float64
elif "32" in str(cfg["memory"]["precision"]):
    dtype = torch.float32
elif "bf16" in str(cfg["memory"]["precision"]):
    dtype = torch.bfloat16
elif "16" in str(cfg["memory"]["precision"]):
    dtype = torch.float16
data_mean = low_snr.mean(mean_std_dims, keepdims=True).to(dtype)
data_std = low_snr.std(mean_std_dims, keepdims=True).to(dtype)

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
if cfg["train-parameters"]["patience"] is not None:
    callbacks = [
        EarlyStopping(patience=cfg["train-parameters"]["patience"], monitor="elbo/val")
    ]
else:
    callbacks = []
trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=cfg["memory"]["gpu"],
    max_epochs=cfg["train-parameters"]["max-epochs"],
    max_time=cfg["train-parameters"]["max-time"],
    callbacks=callbacks,
    plugins=[LightningEnvironment()],
    precision=cfg["memory"]["precision"],
)
# Train model
try:
    trainer.fit(hub, datamodule=datamodule)
except KeyboardInterrupt:
    print("KeyboardInterupt")
finally:
    # Save trained model
    trainer.save_checkpoint(os.path.join(checkpoint_path, f"final_model.ckpt"))
    with open(os.path.join(checkpoint_path, "training-config.yaml"), "w") as f:
    # Save hyperparameters to load models again later
        yaml.dump(cfg, f, default_flow_style=False)
