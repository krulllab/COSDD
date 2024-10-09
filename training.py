import yaml
import os
import argparse

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
parser.add_argument("--config_file", type=str, help=".yaml configuration file") 
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
print(f"Noisy data size: {low_snr.size()}")

if cfg["data"]["clip-outliers"]:
    clip_min = np.percentile(low_snr, 1)
    clip_max = np.percentile(low_snr, 90)
    clip_min_max = (clip_min, clip_max)
else:
    clip_min_max = None

print(f"Effective batch size: {cfg["train-params"]["batch-size"] * cfg["train-params"]["number-grad-batches"]}")
n_iters = np.prod(low_snr.shape[-cfg["data"]["number-dimensions"]:]) // np.prod(cfg["train-params"]["crop-size"])
transform = utils.RandomCrop(cfg["train-params"]["crop-size"])

low_snr = low_snr[torch.randperm(len(low_snr))]
train_set = low_snr[: int(len(low_snr) * cfg["train-params"]["training-split"])]
val_set = low_snr[int(len(low_snr) * cfg["train-params"]["training-split"]) :]

train_set = utils.TrainDataset(train_set, n_iters=n_iters, transform=transform)
val_set = utils.TrainDataset(val_set, n_iters=n_iters, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=cfg["train-params"]["batch-size"], shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=cfg["train-params"]["batch-size"], shuffle=False,
)

z_dims = [cfg["hyper-parameters"]["s-code-channels"] // 2] * cfg["hyper-parameters"]["number-layers"]
downsampling = [0, 1] * (cfg["hyper-parameters"]["number-layers"] // 2)
lvae = LadderVAE(
    colour_channels=low_snr.shape[1],
    img_size=cfg["train-params"]["crop-size"],
    s_code_channels=cfg["hyper-parameters"]["s-code-channels"],
    n_filters=cfg["hyper-parameters"]["s-code-channels"],
    z_dims=z_dims,
    downsampling=downsampling,
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

if cfg["use-direct-denoiser"]:
    direct_denoiser = UNet(
        colour_channels=low_snr.shape[1],
        n_filters=cfg["hyper-parameters"]["s-code-channels"],
        n_layers=cfg["hyper-parameters"]["number-layers"],
        downsampling=downsampling,
        loss_fn=cfg["hyper-parameters"]["direct-denoiser-loss"],
        dimensions=cfg["data"]["number-dimensions"],
    )
else:
    direct_denoiser = None

hub = Hub(
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
    data_mean=low_snr.mean(),
    data_std=low_snr.std(),
    n_grad_batches=cfg["train-params"]["number-grad-batches"],
    checkpointed=cfg["checkpointed"],
    clip_min_max=clip_min_max,
)

checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
logger = TensorBoardLogger(checkpoint_path)

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=cfg["gpus"],
    max_epochs=cfg["train-params"]["max-epochs"],
    log_every_n_steps=len(train_set) // cfg["train-params"]["batch-size"],
    callbacks=[EarlyStopping(patience=cfg["train-params"]["patience"], monitor="val/elbo")],
    precision=cfg["precision"],
)

trainer.fit(hub, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(checkpoint_path, "final_model.ckpt"))
