from pathlib import Path
import yaml
import os
import argparse
import time
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import torch
from pytorch_lightning.plugins.environments import LightningEnvironment
import pytorch_lightning as pl
from tqdm import tqdm
import tifffile

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
cfg = utils.get_defaults(cfg, predict=True)

# Load training configurations that were stored with checkpoint
# Used to load models with correct hyperparameters
checkpoint_path = os.path.join("checkpoints", cfg["model-name"])
with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

print("Loading data...")
low_snr, original_sizes, file_names = utils.load_data(
    paths=cfg["data"]["paths"],
    patterns=cfg["data"]["patterns"],
    axes=cfg["data"]["axes"],
    n_dimensions=cfg["data"]["number-dimensions"],
    return_file_names=True,
)
low_snr_original_shapes = [l.shape for l in low_snr]
if cfg["data"]["patch-size"] is not None:
    # Split data into non-overlapping patches
    low_snr = utils.patchify(low_snr, patch_size=cfg["data"]["patch-size"])

if cfg["data"]["clip-outliers"]:
    # Clip data values outside of 1st and 99th percentiles
    print("Clippping min...")
    clip_min = utils.percentile(low_snr, 1)
    print("Clippping max...")
    clip_max = utils.percentile(low_snr, 99)
    low_snr = [torch.clamp(l, clip_min, clip_max) for l in low_snr]

# Use data to create pytorch dataset
predict_set = utils.PredictDataset(low_snr)

# Use dataset to create pytorch dataloader
predict_loader = torch.utils.data.DataLoader(
    predict_set,
    batch_size=cfg["predict-parameters"]["batch-size"],
    shuffle=False,
)

# Load models with trained parameters
lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr[0].shape[0])

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
# TODO: does this work for all possible scenarios?
if cfg["use-direct-denoiser"]:
    assert hub.direct_denoiser is not None, "Direct denoiser not trained. Set `use-direct-denoiser: False`."
    # If the direct denoiser was trained, uses it for inference
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
    denoised = predictor.predict(hub, predict_loader)
    denoised = torch.cat(denoised, dim=0)
else:
    # If direct denoiser was not trained, randomly sample solutions and average them
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
    for _ in tqdm(range(cfg["n-samples"])):
        out = predictor.predict(hub, predict_loader)
        out = torch.cat(out, dim=0)
        samples.append(out)

    samples = torch.stack(samples, dim=1)
    denoised = torch.mean(samples, dim=1)
if denoised.dtype == torch.bfloat16:
    # bfloat16 can't be saved as tiff, so switches to float32
    denoised = denoised.float()
if cfg["data"]["patch-size"] is not None:
    # If data was patched into non-overlapping windows, restore original shape.
    denoised = utils.unpatchify(denoised, original_shapes=low_snr_original_shapes, patch_size=cfg["data"]["patch-size"])
denoised = [d.numpy() for d in denoised]

# Restore dimensions to how they were stored before converting to pytorch [S, C, Z | Y | X]
denoised = utils.SCZYX_to_axes(
    denoised, original_axes=cfg["data"]["axes"], original_sizes=original_sizes, n_dimensions=cfg["data"]["number-dimensions"]
)

# Saves denoised images using date and time as directory name
# Each image that was an individual file before denoising will be saved as an individual denoised image
current_time = time.strftime("%d-%m-%y_%H-%M-%S", time.localtime())
save_path = Path(os.path.join(cfg["data"]["save-path"], f"denoised-{current_time}"))
if not os.path.exists(save_path):
    print(f"Creating directory: {save_path}")
    os.makedirs(save_path)
print(f"Saving denoised images to {save_path}")
for i, image in enumerate(denoised):
    file_name = file_names[i]
    save_file_name = save_path / (str(file_name) + ".tif")
    save_file_name.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_file_name, image)
