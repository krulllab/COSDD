import yaml
import os
import argparse
import time
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import torch
from pytorch_lightning.plugins.environments import LightningEnvironment
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import tifffile

import utils
from models.get_models import get_models
from models.hub import Hub


assert torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

with open(args.config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = utils.get_defaults(cfg, predict=True)

checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

print("Loading data...")
low_snr = utils.load_data(
    cfg["data"]["paths"],
    cfg["data"]["patterns"],
    cfg["data"]["axes"],
    cfg["data"]["number-dimensions"],
)
original_shape = low_snr.shape
if cfg["data"]["patch-size"] is not None:
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])
print(f"Noisy data shape: {low_snr.size()}")

if cfg["data"]["clip-outliers"]:
    print("Clippping min...")
    clip_min = np.percentile(low_snr, 1)
    print("Clippping max...")
    clip_max = np.percentile(low_snr, 99)
    low_snr = torch.clamp(low_snr, clip_min, clip_max)

predict_set = utils.PredictDataset(low_snr)

predict_loader = torch.utils.data.DataLoader(
    predict_set,
    batch_size=cfg["predict-parameters"]["batch-size"],
    shuffle=False,
)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr.shape[1])

checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
hub = Hub.load_from_checkpoint(
    os.path.join(checkpoint_path, "final_model.ckpt"),
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
)

if isinstance(cfg["memory"]["gpu"], int):
    cfg["memory"]["gpu"] = [cfg["memory"]["gpu"]]
if direct_denoiser is not None:
    hub.direct_pred = True
    predictor = pl.Trainer(
        accelerator="gpu",
        devices=cfg["memory"]["gpu"],
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        precision=cfg["memory"]["precision"],
        plugins=[LightningEnvironment()],
    )
    direct = predictor.predict(hub, predict_loader)
    denoised = torch.cat(direct, dim=0)
else:
    hub.direct_pred = False
    predictor = pl.Trainer(
        accelerator="gpu",
        devices=cfg["memory"]["gpu"],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        plugins=[LightningEnvironment()],
        precision=cfg["memory"]["precision"],
    )
    samples = []
    for _ in tqdm(range(cfg["n_samples"])):
        out = predictor.predict(hub, predict_loader)
        out = torch.cat(out, dim=0)
        samples.append(out)

    samples = torch.stack(samples, dim=1)
    denoised = torch.mean(samples, dim=1)
if denoised.shape != original_shape:
    denoised = utils.unpatchify(
        denoised, original_shape=original_shape, patch_size=cfg["data"]["patch-size"]
    )

if not os.path.exists(cfg["data"]["save-path"]):
    print(f'Creating directory: {cfg["data"]["save-path"]}')
    os.makedirs(cfg["data"]["save-path"])
current_time = time.strftime("%d-%m-%y-%X", time.localtime())
save_path = os.path.join(cfg["data"]["save-path"], f"denoised-{current_time}.tif")
print(f"Saving denoised images to {save_path}")
tifffile.imwrite(save_path, denoised.numpy())
