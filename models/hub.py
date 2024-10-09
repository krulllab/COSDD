import torch
import pytorch_lightning as pl
from torch import optim
import random
import numpy as np
import matplotlib.pyplot as plt

from .utils import plot_to_image


class Hub(pl.LightningModule):
    """Hub to unify the training of the VAE, signal decoder and direct denoiser.


    Parameters
    ----------
    vae : torch.nn.Module
        A VAE that outputs a latent code.
    ar_decoder : torch.nn.Module
        A decoder that takes the latent code and outputs the parameters of a
        predictive distribution for the input.
    s_decoder : torch.nn.Module
        A decoder that takes the latent code and maps it into image space.
    data_mean : float, optional
        The mean of the training data. Used to normalise the data.
    data_std : float, optional
        The standard deviation of the training data. Used to normalise the data.
    n_grad_batches : int, optional
        The number of batches to accumulate gradients over before updating the
        weights.
    checkpointed : bool, optional
        Whether to use activation checkpointing in the forward pass.
    clip_min_max : list, optional
        Clips pixel intensities to be within these values. Useful if training
        crashes due to NaNs.
    """

    def __init__(
        self,
        vae,
        ar_decoder,
        s_decoder,
        direct_denoiser=None,
        data_mean=0,
        data_std=1,
        n_grad_batches=1,
        checkpointed=True,
        clip_min_max=None,
    ):
        self.save_hyperparameters(
            ignore=["vae", "ar_decoder", "s_decoder", "direct_denoiser"]
        )

        super().__init__()
        self.vae = vae
        self.ar_decoder = ar_decoder
        self.s_decoder = s_decoder
        self.direct_denoiser = direct_denoiser
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_grad_batches = n_grad_batches
        self.clip_min_max = clip_min_max

        if hasattr(vae, "checkpointed"):
            vae.checkpointed = checkpointed
        if hasattr(ar_decoder, "checkpointed"):
            ar_decoder.checkpointed = checkpointed
        if hasattr(s_decoder, "checkpointed"):
            s_decoder.checkpointed = checkpointed
        if direct_denoiser is not None:
            if hasattr(direct_denoiser, "checkpointed"):
                direct_denoiser.checkpointed = checkpointed

        self.automatic_optimization = False

        self.direct_pred = False  # Whether to use direct denoiser for prediction

    def vae_forward(self, x):
        if self.clip_min_max is not None:
            x = torch.clamp(
                x, min=self.clip_min_max[0], max=self.clip_min_max[1]
            )
        vae_out = self.vae(x)
        s_code = vae_out["s_code"]

        x_params = self.ar_decoder(x, s_code)
        out = {
            "s_code": s_code,
            "q_list": vae_out["q_list"],
            "p_list": vae_out["p_list"],
            "x_params": x_params,
        }
        return out

    def s_decoder_forward(self, s_code):
        s_hat = self.s_decoder(s_code.detach())
        return s_hat

    def direct_denoiser_forward(self, x):
        if self.clip_min_max is not None:
            x = torch.clamp(
                x, min=self.clip_min_max[0], max=self.clip_min_max[1]
            )
        s_direct = self.direct_denoiser(x)
        return s_direct

    def get_optimizer_scheduler(self, params):
        optimizer = optim.Adamax(params)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=50
        )
        return optimizer, scheduler

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        vae_params = list(self.vae.parameters()) + list(self.ar_decoder.parameters())
        vae_optimizer, vae_scheduler = self.get_optimizer_scheduler(vae_params)
        optimizers.append(vae_optimizer)
        schedulers.append(vae_scheduler)

        sd_params = self.s_decoder.parameters()
        sd_optimizer, sd_scheduler = self.get_optimizer_scheduler(sd_params)
        optimizers.append(sd_optimizer)
        schedulers.append(sd_scheduler)

        if self.direct_denoiser is not None:
            dd_params = self.direct_denoiser.parameters()
            dd_optimizer, dd_scheduler = self.get_optimizer_scheduler(dd_params)
            optimizers.append(dd_optimizer)
            schedulers.append(dd_scheduler)

        return optimizers, schedulers

    def vae_loss(self, x, out):
        if self.clip_min_max is not None:
            x = torch.clamp(
                x, min=self.clip_min_max[0], max=self.clip_min_max[1]
            )
        losses = {}
        kl_div = self.vae.kl_divergence(out["q_list"], out["p_list"])
        kl_div = kl_div / x.numel()
        nll = -self.ar_decoder.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_div + nll
        losses["elbo"] = elbo
        losses["kl_div"] = kl_div
        losses["nll"] = nll
        return losses

    def s_decoder_loss(self, x, s_hat):
        sd_loss = self.s_decoder.loss(x, s_hat).mean()
        return sd_loss

    def direct_denoiser_loss(self, s_hat, s_direct):
        dd_loss = self.direct_denoiser.loss(s_hat.detach(), s_direct).mean()
        return dd_loss

    def training_step(self, batch, batch_idx):
        x = (batch - self.data_mean) / self.data_std
        vae_out = self.vae_forward(x)
        vae_losses = self.vae_loss(x, vae_out)
        self.manual_backward(vae_losses["elbo"])
        self.log("train/elbo", vae_losses["elbo"], on_step=False, on_epoch=True)
        self.log("train/kl_div", vae_losses["kl_div"], on_step=False, on_epoch=True)
        self.log("train/nll", vae_losses["nll"], on_step=False, on_epoch=True)
        if (batch_idx + 1) % self.n_grad_batches == 0:
            optimizer = self.optimizers()[0]
            optimizer.step()
            optimizer.zero_grad()

        s_hat = self.s_decoder_forward(vae_out["s_code"])
        sd_loss = self.s_decoder_loss(x, s_hat)
        self.manual_backward(sd_loss)
        self.log("train/sd_loss", sd_loss, on_step=False, on_epoch=True)
        if (batch_idx + 1) % self.n_grad_batches == 0:
            optimizer = self.optimizers()[1]
            optimizer.step()
            optimizer.zero_grad()

        if self.direct_denoiser is not None:
            s_direct = self.direct_denoiser_forward(x)
            dd_loss = self.direct_denoiser_loss(s_hat, s_direct)
            self.manual_backward(dd_loss)
            self.log("train/dd_loss", dd_loss, on_step=False, on_epoch=True)
            if (batch_idx + 1) % self.n_grad_batches == 0:
                optimizer = self.optimizers()[2]
                optimizer.step()
                optimizer.zero_grad()

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        vae_scheduler = schedulers[0]
        vae_scheduler.step(self.trainer.callback_metrics["val/elbo"])

        sd_scheduler = schedulers[1]
        sd_scheduler.step(self.trainer.callback_metrics["val/sd_loss"])

        if self.direct_denoiser is not None:  # and self.current_epoch > 50:
            dd_scheduler = schedulers[2]
            dd_scheduler.step(self.trainer.callback_metrics["val/dd_loss"])

    def log_image(self, img, img_name):
        normalised_img = (img - np.percentile(img, 1)) / (
            np.percentile(img, 99) - np.percentile(img, 1)
        )
        clamped_img = np.clip(normalised_img, 0, 1)
        self.trainer.logger.experiment.add_image(
            img_name, clamped_img, self.current_epoch
        )

    def log_val_images(self, batch, samples, mmse, direct):
        n_channels = batch.shape[1]
        d = batch.ndim - 2
        if d == 1:
            # Make 1D plots
            for ch in range(n_channels):
                figure = plt.figure()
                plt.plot(batch[0, ch].cpu().float(), label="Noisy", color="blue")
                for j in range(samples.size(0)):
                    plt.plot(samples[j][ch].cpu().float(), color="orange", alpha=0.5)
                plt.plot(mmse[0, ch].cpu().float(), label="Denoised", color="orange")
                if direct is not None:
                    plt.plot(direct[0, ch].cpu().float(), label="Direct", color="green")
                plt.legend()
                img = plot_to_image(figure)
                self.log_image(img, f"channel_{ch}")
                plt.close(figure)
        else:
            if d == 3:
                # Show the first slice of z-stack as image
                batch = batch[:, :, 0]
                samples = samples[:, :, 0]
                mmse = mmse[:, :, 0]
                if direct is not None:
                    direct = direct[:, :, 0]
            # Show 2D images. Only shows first 3 channels
            self.log_image(batch[0, :3].cpu().float().numpy(), "inputs/noisy")
            self.log_image(samples[0, :3].cpu().float().numpy(), "outputs/sample 1")
            self.log_image(samples[1, :3].cpu().float().numpy(), "outputs/sample 2")
            self.log_image(
                mmse[0, :3].cpu().float().numpy(), "outputs/mmse (10 samples)"
            )
            if direct is not None:
                self.log_image(
                    direct[0, :3].cpu().float().numpy(), "outputs/direct estimate"
                )

    def validation_step(self, batch, batch_idx):
        x = (batch - self.data_mean) / self.data_std
        vae_out = self.vae_forward(x)
        vae_losses = self.vae_loss(x, vae_out)
        self.log("val/elbo", vae_losses["elbo"], on_step=False, on_epoch=True)
        self.log("val/kl_div", vae_losses["kl_div"], on_step=False, on_epoch=True)
        self.log("val/nll", vae_losses["nll"], on_step=False, on_epoch=True)

        s_hat = self.s_decoder_forward(vae_out["s_code"])
        sd_loss = self.s_decoder_loss(x, s_hat)
        self.log("val/sd_loss", sd_loss, on_step=False, on_epoch=True)

        if self.direct_denoiser is not None:
            s_direct = self.direct_denoiser_forward(x)
            dd_loss = self.direct_denoiser_loss(s_hat, s_direct)
            self.log("val/dd_loss", dd_loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            idx = random.randint(0, x.shape[0] - 1)
            vae_out = self.vae_forward(x[idx : idx + 1].repeat_interleave(10, 0))
            s_hat = self.s_decoder_forward(vae_out["s_code"])
            mmse = torch.mean(s_hat, 0, keepdim=True)
            if self.direct_denoiser is not None:
                s_direct = self.direct_denoiser_forward(x[idx : idx + 1])
            else:
                s_direct = None
            self.log_val_images(x[idx : idx + 1], s_hat, mmse, s_direct)

    def predict_step(self, batch, _):
        self.eval()
        x = (batch - self.data_mean) / self.data_std
        if self.direct_pred == True:
            if self.direct_denoiser is not None:
                s_hat = self.direct_denoiser(x)
            else:
                print(
                    "Direct denoiser not trained. Set use_direct_denoiser in training script to True."
                )
                return
        else:
            vae_out = self.vae(x)
            s_code = vae_out["s_code"]
            s_hat = self.s_decoder(s_code)

        return s_hat * self.data_std + self.data_mean

    @torch.no_grad()
    def sample_prior(self, n_imgs):
        self.eval()
        self.vae.mode_pred = True
        s_code = self.vae.sample_from_prior(n_imgs)
        s = self.s_decoder(s_code)
        s = s * self.data_std + self.data_mean
        x = self.ar_decoder.sample(s_code)
        x = x * self.data_std + self.data_mean

        return {"x": x, "s": s}

    @torch.no_grad()
    def reconstruct(self, x):
        self.eval()
        x = (x - self.data_mean) / self.data_std
        s_code = self.vae(x)["s_code"]
        s_hat = self.s_decoder(s_code)
        x_hat = self.ar_decoder.sample(s_code)
        s_hat = s_hat * self.data_std + self.data_mean
        x_hat = x_hat * self.data_std + self.data_mean

        return {"x_hat": x_hat, "s_hat": s_hat}
