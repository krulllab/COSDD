import math

from .lvae import LadderVAE
from .pixelcnn import PixelCNN
from .s_decoder import SDecoder
from .unet import UNet


def get_models(config, n_channels):
    z_dims = [config["hyper-parameters"]["s-code-channels"] // 2] * config[
        "hyper-parameters"
    ]["number-layers"]
    min_size = min(config["train-parameters"]["crop-size"])
    num_halves = math.floor(math.log2(min_size)) - 1
    downsampling = [1] * config["hyper-parameters"]["number-layers"]
    difference = max(config["hyper-parameters"]["number-layers"] - num_halves, 0)
    i = 0
    while difference > 0:
        for j in range(config["hyper-parameters"]["number-layers"] // 2):
            downsampling[i + j * 2] = 0
            difference -= 1
            if difference == 0:
                break
        i += 1

    lvae = LadderVAE(
        colour_channels=n_channels,
        img_size=config["train-parameters"]["crop-size"],
        s_code_channels=config["hyper-parameters"]["s-code-channels"],
        n_filters=config["hyper-parameters"]["s-code-channels"],
        z_dims=z_dims,
        downsampling=downsampling,
        monte_carlo_kl=config["train-parameters"]["monte-carlo-kl"],
        dimensions=config["data"]["number-dimensions"],
    )

    ar_decoder = PixelCNN(
        colour_channels=n_channels,
        s_code_channels=config["hyper-parameters"]["s-code-channels"],
        kernel_size=5,
        noise_direction=config["hyper-parameters"]["noise-direction"],
        n_filters=64,
        n_layers=4,
        n_gaussians=config["hyper-parameters"]["number-gaussians"],
        dimensions=config["data"]["number-dimensions"],
    )

    s_decoder = SDecoder(
        colour_channels=n_channels,
        s_code_channels=config["hyper-parameters"]["s-code-channels"],
        n_filters=config["hyper-parameters"]["s-code-channels"],
        dimensions=config["data"]["number-dimensions"],
    )

    if config["train-parameters"]["use-direct-denoiser"]:
        direct_denoiser = UNet(
            colour_channels=n_channels,
            n_filters=config["hyper-parameters"]["s-code-channels"],
            n_layers=config["hyper-parameters"]["number-layers"],
            downsampling=downsampling,
            loss_fn=config["train-parameters"]["direct-denoiser-loss"],
            dimensions=config["data"]["number-dimensions"],
        )
    else:
        direct_denoiser = None

    return lvae, ar_decoder, s_decoder, direct_denoiser
