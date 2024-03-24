import torch
import pytorch_lightning as pl
from torch import optim
import random
import numpy as np


class DVLAE(pl.LightningModule):
    """Module to unify the training of the VAE and direct denoiser.


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
    ):
        self.save_hyperparameters()

        super().__init__()
        self.vae = vae
        self.ar_decoder = ar_decoder
        self.s_decoder = s_decoder
        self.direct_denoiser = direct_denoiser
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_grad_batches = n_grad_batches

        self.automatic_optimization = False

        self.direct_pred = False  # Whether to use direct denoiser for prediction

    def forward(self, x):
        x = (x - self.data_mean) / self.data_std

        vae_out = self.vae(x)
        s_code = vae_out["s_code"]

        x_params = self.ar_decoder(x, s_code)

        s_hat = self.s_decoder(s_code.detach())

        if self.direct_denoiser is not None:# and self.current_epoch > 50:
            s_direct = self.direct_denoiser(x)
        else:
            s_direct = None

        out = {
            "s_code": s_code,
            "kl_loss": vae_out["kl_loss"],
            "x_params": x_params,
            "s_hat": s_hat,
            "s_direct": s_direct,
        }

        return out

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        vae_params = list(self.vae.parameters()) + list(self.ar_decoder.parameters())
        vae_optimizer = optim.Adamax(vae_params)
        optimizers.append(vae_optimizer)
        vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, factor=0.1, patience=50
        )
        schedulers.append(vae_scheduler)

        sd_params = self.s_decoder.parameters()
        sd_optimizer = optim.Adamax(sd_params)
        optimizers.append(sd_optimizer)
        sd_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            sd_optimizer, factor=0.1, patience=50
        )
        schedulers.append(sd_scheduler)

        if self.direct_denoiser is not None:
            dd_params = self.direct_denoiser.parameters()
            dd_optimizer = optim.Adamax(dd_params)
            optimizers.append(dd_optimizer)
            dd_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                dd_optimizer, factor=0.1, patience=50
            )
            schedulers.append(dd_scheduler)

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        self.vae.mode_pred = False
        out = self(batch)

        x = (batch - self.data_mean) / self.data_std

        kl_loss = out["kl_loss"]
        reconstruction_loss = -self.ar_decoder.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_loss + reconstruction_loss
        self.manual_backward(elbo)
        self.log("train/elbo", elbo)
        self.log("train/kl_loss", kl_loss)
        self.log("train/reconstruction_loss", reconstruction_loss)

        sd_loss = self.s_decoder.loss(x, out["s_hat"]).mean()
        self.manual_backward(sd_loss)
        self.log("train/sd_loss", sd_loss)

        if out["s_direct"] is not None:
            dd_loss = self.direct_denoiser.loss(
                out["s_hat"].detach(), out["s_direct"]
            ).mean()
            self.manual_backward(dd_loss)
            self.log("train/dd_loss", dd_loss)

        if (batch_idx + 1) % self.n_grad_batches == 0:
            optimizers = self.optimizers()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        vae_scheduler = schedulers[0]
        vae_scheduler.step(self.trainer.callback_metrics["val/elbo"])

        sd_scheduler = schedulers[1]
        sd_scheduler.step(self.trainer.callback_metrics["val/sd_loss"])

        if self.direct_denoiser is not None:# and self.current_epoch > 50:
            dd_scheduler = schedulers[2]
            dd_scheduler.step(self.trainer.callback_metrics["val/dd_loss"])

    def log_tensorboard_images(self, img, img_name):
        img = img.cpu().numpy()
        normalised_img = (img - np.percentile(img, 1)) / (
            np.percentile(img, 99) - np.percentile(img, 1)
        )
        clamped_img = np.clip(normalised_img, 0, 1)
        self.trainer.logger.experiment.add_image(
            img_name, clamped_img, self.current_epoch
        )

    def validation_step(self, batch, batch_idx):
        self.vae.mode_pred = False
        out = self(batch)

        x = (batch - self.data_mean) / self.data_std

        kl_loss = out["kl_loss"]
        reconstruction_loss = -self.ar_decoder.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_loss + reconstruction_loss
        self.log("val/elbo", elbo)
        self.log("val/kl_loss", kl_loss)
        self.log("val/reconstruction_loss", reconstruction_loss)

        sd_loss = self.s_decoder.loss(x, out["s_hat"]).mean()
        self.log("val/sd_loss", sd_loss)

        if out["s_direct"] is not None:
            dd_loss = self.direct_denoiser.loss(
                out["s_hat"].detach(), out["s_direct"]
            ).mean()
            self.log("val/dd_loss", dd_loss)

        if batch_idx == 0:
            idx = random.randint(0, batch.shape[0] - 1)
            out = self.forward(batch[idx : idx + 1].repeat_interleave(10, 0))
            mmse = torch.mean(out["s_hat"], 0, keepdim=True)
            self.log_tensorboard_images(batch[idx], "inputs/noisy")
            self.log_tensorboard_images(out["s_hat"][0], "outputs/sample 1")
            self.log_tensorboard_images(out["s_hat"][1], "outputs/sample 2")
            self.log_tensorboard_images(mmse[0], "outputs/mmse (10 samples)")
            if out["s_direct"] is not None:
                self.log_tensorboard_images(
                    out["s_direct"][0], "outputs/direct estimate"
                )

    def predict_step(self, batch, _):
        self.eval()
        self.vae.mode_pred = True
        x = (batch - self.data_mean) / self.data_std
        if self.direct_pred == True:
            s_hat = self.direct_denoiser(x)
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
        self.vae.mode_pred = True
        x = (x - self.data_mean) / self.data_std
        s_code = self.vae(x)["s_code"]
        s_hat = self.s_decoder(s_code)
        x_hat = self.ar_decoder.sample(s_code)
        s_hat = s_hat * self.data_std + self.data_mean
        x_hat = x_hat * self.data_std + self.data_mean

        return {"x_hat": x_hat, "s_hat": s_hat}
