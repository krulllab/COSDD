import os
import urllib

import torch
from torchvision import transforms
import tifffile
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lvae.models.lvae import LadderVAE
from ar_decoder.pixelcnn import PixelCNN
from s_decoder import SDecoder
from direct_denoiser.models.unet import UNet
from dvlae import DVLAE


use_cuda = torch.cuda.is_available()

# create a folder for our data.
if not os.path.exists("./data"):
    os.mkdir("./data")

# check if data has been downloaded already
lowsnr_path = "data/mito-confocal-lowsnr.tif"
if not os.path.exists(lowsnr_path):
    urllib.request.urlretrieve(
        "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100888/03-mito-confocal/mito-confocal-lowsnr.tif",
        lowsnr_path,
    )
highsnr_path = "data/mito-confocal-highsnr.tif"
if not os.path.exists(highsnr_path):
    urllib.request.urlretrieve(
        "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100888/03-mito-confocal/mito-confocal-highsnr.tif",
        highsnr_path,
    )

# load the data
low_snr = tifffile.imread(lowsnr_path).astype(float)
low_snr = torch.from_numpy(low_snr).to(torch.float32)[:, None]


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


batch_size = 4
crop_size = 256
train_split = 0.9

n_iters = (low_snr[0].shape[-1] * low_snr[0].shape[-2]) // crop_size**2
transform = transforms.RandomCrop(crop_size)

low_snr = low_snr[torch.randperm(len(low_snr))]
train_set = low_snr[: int(len(low_snr) * train_split)]
val_set = low_snr[int(len(low_snr) * train_split) :]

train_set = TrainDataset(train_set, n_iters=n_iters, transform=transform)
val_set = TrainDataset(val_set, n_iters=n_iters, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, pin_memory=True
)

s_code_channels = 64  # Set this to 128 to use the full size model

n_layers = 6  # Set this to 14 to use the full size model
z_dims = [s_code_channels // 2] * n_layers
downsampling = [
    1
] * n_layers  # Set this to [0, 1] * (n_layers // 2) when using the full size model
lvae = LadderVAE(
    colour_channels=low_snr.shape[1],
    img_shape=(crop_size, crop_size),
    s_code_channels=s_code_channels,
    n_filters=s_code_channels,
    z_dims=z_dims,
    downsampling=downsampling,
)

ar_decoder = PixelCNN(
    colour_channels=low_snr.shape[1],
    s_code_channels=s_code_channels,
    kernel_size=5,
    RF_shape="horizontal",
    n_filters=64,
    n_layers=4,
    n_gaussians=7,
)

s_decoder = SDecoder(
    colour_channels=low_snr.shape[1],
    s_code_channels=s_code_channels,
    n_filters=s_code_channels,
)

use_direct_denoiser = True
if use_direct_denoiser:
    direct_denoiser = UNet(
        colour_channels=low_snr.shape[1],
        img_shape=(crop_size, crop_size),
        s_code_channels=low_snr.shape[1],
        n_filters=s_code_channels,
        n_layers=n_layers,
        downsampling=downsampling,
        loss_fn="L2",
    )
else:
    direct_denoiser = None

dvlae = DVLAE(
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
    data_mean=low_snr.mean(),
    data_std=low_snr.std(),
    n_grad_batches=4,
)

model_name = "mito-confocal"
checkpoint_path = os.path.join("checkpoints", model_name)
logger = TensorBoardLogger(checkpoint_path)

max_epochs = 1000  # Set this to 1000 for full training. Here we train for 10 epochs for demonstration purposes
patience = 100  # Set this to 100 for full training. Here we use a small value for demonstration purposes

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu" if use_cuda else "cpu",
    devices=1,
    max_epochs=max_epochs,
    log_every_n_steps=len(train_set) // batch_size,
    callbacks=[EarlyStopping(patience=patience, monitor="val/elbo")],
)

trainer.fit(dvlae, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(checkpoint_path, "final_model.ckpt"))
