import math

from .lvae import LadderVAE
from .pixelcnn import PixelCNN
from .s_decoder import SDecoder
from .unet import UNet


def get_models(config, n_channels):
    z_dims = [config["hyper-parameters"]["s-code-channels"] // 2] * config[
        "hyper-parameters"
    ]["number-layers"]
    find_num_halves = lambda x: math.floor(math.log2(x)) - 1
    num_halves = [find_num_halves(s) for s in config["train-parameters"]["crop-size"]]
    find_difference = lambda nh: max(config["hyper-parameters"]["number-layers"] - nh, 0)
    difference = [find_difference(nh) for nh in num_halves]
    n_dim = len(config["train-parameters"]["crop-size"])
    downsampling = [[1 for _ in range(n_dim)] for _ in range(config["hyper-parameters"]["number-layers"])]
    for i in range(n_dim):
        j = 0
        while difference[i] > 0:
            for k in range(config["hyper-parameters"]["number-layers"] // 2):
                downsampling[j + k * 2][i] = 0
                difference[i] -= 1
                if difference[i] == 0:
                    break
            j += 1

    lvae = LadderVAE(
        colour_channels=n_channels,
        img_size=config["train-parameters"]["crop-size"],
        s_code_channels=config["hyper-parameters"]["s-code-channels"],
        n_filters=config["hyper-parameters"]["s-code-channels"],
        scale_initialisation=config["hyper-parameters"]["scale-initialisation"],
        z_dims=z_dims,
        downsampling=downsampling,
        dimensions=config["data"]["number-dimensions"],
    )

    ar_decoder = PixelCNN(
        colour_channels=n_channels,
        s_code_channels=config["hyper-parameters"]["s-code-channels"],
        kernel_size=5,
        noise_direction=config["hyper-parameters"]["noise-direction"],
        n_filters=64,
        n_layers=4,
        n_components=config["hyper-parameters"]["number-components"],
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
