import yaml
import os
import argparse
import math
import pickle
import time
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import tifffile

import utils
from models.lvae import LadderVAE
from models.pixelcnn import PixelCNN
from models.s_decoder import SDecoder
from models.unet import UNet
from models.hub import Hub


assert torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

with open(args.config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = utils.get_defaults(cfg, predict=True)

checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
with open(os.path.join(checkpoint_path, "training-config.pkl"), "rb") as f:
    train_cfg = pickle.load(f)

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

z_dims = [train_cfg["hyper-parameters"]["s-code-channels"] // 2] * train_cfg[
    "hyper-parameters"
]["number-layers"]
max_size = max(train_cfg["train-parameters"]["crop-size"])
num_halves = math.floor(math.log2(max_size)) - 1
downsampling = [1] * train_cfg["hyper-parameters"]["number-layers"]
difference = max(train_cfg["hyper-parameters"]["number-layers"] - num_halves, 0)
i = 0
while difference > 0:
    for j in range(train_cfg["hyper-parameters"]["number-layers"] // 2):
        downsampling[i + j * 2] = 0
        difference -= 1
        if difference == 0:
            break
    i += 1

lvae = LadderVAE(
    colour_channels=low_snr.shape[1],
    img_size=train_cfg["train-parameters"]["crop-size"],
    s_code_channels=train_cfg["hyper-parameters"]["s-code-channels"],
    n_filters=train_cfg["hyper-parameters"]["s-code-channels"],
    z_dims=z_dims,
    downsampling=downsampling,
    monte_carlo_kl=train_cfg["train-parameters"]["monte-carlo-kl"],
    dimensions=train_cfg["data"]["number-dimensions"],
)

ar_decoder = PixelCNN(
    colour_channels=low_snr.shape[1],
    s_code_channels=train_cfg["hyper-parameters"]["s-code-channels"],
    kernel_size=5,
    noise_direction=train_cfg["hyper-parameters"]["noise-direction"],
    n_filters=64,
    n_layers=4,
    n_gaussians=train_cfg["hyper-parameters"]["number-gaussians"],
    dimensions=train_cfg["data"]["number-dimensions"],
)

s_decoder = SDecoder(
    colour_channels=low_snr.shape[1],
    s_code_channels=train_cfg["hyper-parameters"]["s-code-channels"],
    n_filters=train_cfg["hyper-parameters"]["s-code-channels"],
    dimensions=train_cfg["data"]["number-dimensions"],
)

if train_cfg["train-parameters"]["use-direct-denoiser"]:
    direct_denoiser = UNet(
        colour_channels=low_snr.shape[1],
        n_filters=train_cfg["hyper-parameters"]["s-code-channels"],
        n_layers=train_cfg["hyper-parameters"]["number-layers"],
        downsampling=downsampling,
        loss_fn=train_cfg["train-parameters"]["direct-denoiser-loss"],
        dimensions=train_cfg["data"]["number-dimensions"],
    )
else:
    direct_denoiser = None

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
    denoised = utils.unpatchify(denoised, original_shape=original_shape, patch_size=cfg["data"]["patch-size"])

if not os.path.exists(cfg["data"]["save-path"]):
    print(f"Creating directory: {cfg["data"]["save-path"]}")
    os.makedirs(cfg["data"]["save-path"])
current_time = time.strftime('%d-%m-%y-%X', time.localtime())
save_path = os.path.join(cfg["data"]["save-path"], f"denoised-{current_time}.tif")
tifffile.imwrite(save_path, denoised.numpy())
