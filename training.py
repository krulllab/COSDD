import yaml
import os
import argparse
import warnings
import math

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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

if "64" in str(cfg["memory"]["precision"]):
    dtype = torch.float64
elif "32" in str(cfg["memory"]["precision"]):
    dtype = torch.float32
elif "bf16" in str(cfg["memory"]["precision"]):
    dtype = torch.bfloat16
elif "16" in str(cfg["memory"]["precision"]):
    dtype = torch.float16

print("Loading data...")
low_snr, _ = utils.load_data(
    paths=cfg["data"]["paths"],
    patterns=cfg["data"]["patterns"],
    axes=cfg["data"]["axes"],
    n_dimensions=cfg["data"]["number-dimensions"],
)
if cfg["data"]["patch-size"] is not None:
    # Split data into non-overlapping patches
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])

# The loaded data is split into training and validation sets along the same axis, i.e. different images/volumes
# are used for training validation.
# If there are too few images for the chosen training/validation split (e.g. 0.9/0.1), individual images will
# have to be broken up into patches, and the patches randomly split into training/validation sets.
# This will try to do so automatically, but should be done manually by setting data: patch-size configuration option.
if int(cfg["train-parameters"]["training-split"] * len(low_snr)) == len(low_snr):
    val_split = round(1 - cfg["train-parameters"]["training-split"], 3)
    raise Exception(
        f'Data of length: {len(low_snr)} cannot be split {cfg["train-parameters"]["training-split"]}/\
          {val_split} train/validation along sample axis.'
    )

if cfg["data"]["clip-outliers"]:
    # To avoid outliers causing problems, clip data values outside of 1st and 99th percentiles
    print("Clippping min...")
    clip_min = utils.percentile(low_snr, 1)
    print("Clippping max...")
    clip_max = utils.percentile(low_snr, 99)
    low_snr = [torch.clamp(l, clip_min, clip_max) for l in low_snr]

datamodule = utils.DataModule(
    low_snr=low_snr,
    batch_size=cfg["train-parameters"]["batch-size"],
    rand_crop_size=cfg["train-parameters"]["crop-size"],
    train_split=cfg["train-parameters"]["training-split"],
)
data_max = low_snr.max()
data_min = low_snr.min()
print(f"data min {data_min} data max {data_max}")
# Load models
lvae, ar_decoder, s_decoder, direct_denoiser = get_models(cfg, low_snr[0].shape[0], data_max=data_max, data_min=data_min)

# Each channel is normalised individually.
data_mean, data_std = utils.mean_std(low_snr)

if cfg["pretrained-path"] is not None:
    hub = Hub(
        vae=lvae,
        ar_decoder=ar_decoder,
        s_decoder=s_decoder,
        direct_denoiser=None,
        data_mean=data_mean,
        data_std=data_std,
        n_grad_batches=cfg["train-parameters"]["number-grad-batches"],
        checkpointed=cfg["memory"]["checkpointed"],
    )
    params = torch.load(cfg["pretrained-path"], weights_only=True)
    hub.load_state_dict(params, strict=False)
    hub.direct_denoiser = direct_denoiser
else:
    hub = Hub(
        vae=lvae,
        ar_decoder=ar_decoder,
        s_decoder=s_decoder,
        direct_denoiser=None,
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
    trainer.fit(hub, datamodule=datamodule, ckpt_path=cfg["continue-checkpoint"])
    trainer.fit(hub, datamodule=datamodule, ckpt_path=cfg["continue-checkpoint"])
except KeyboardInterrupt:
    print("KeyboardInterupt")
finally:
    # Save trained model
    trainer.save_checkpoint(os.path.join(checkpoint_path, f"final_model.ckpt"), weights_only=True)
    with open(os.path.join(checkpoint_path, "training-config.yaml"), "w") as f:
    # Save hyperparameters to load models again later
        yaml.dump(cfg, f, default_flow_style=False)
