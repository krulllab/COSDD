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
    noise_model : torch.nn.Module
        A decoder that takes the latent code and outputs the parameters of a
        distribution that the input is sampled from.
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
        noise_model,
        s_decoder,
        data_mean=0,
        data_std=1,
        n_grad_batches=1,
    ):
        self.save_hyperparameters()

        super().__init__()
        self.vae = vae
        self.noise_model = noise_model
        self.s_decoder = s_decoder
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_grad_batches = n_grad_batches

        self.automatic_optimization = False

    def forward(self, x):
        x = (x - self.data_mean) / self.data_std

        vae_out = self.vae(x)
        s_code = vae_out["s_code"]

        x_params = self.noise_model(x, s_code)

        s_hat = self.s_decoder(s_code.detach())

        out = {
            "s_code": s_code,
            "kl_loss": vae_out["kl_loss"],
            "x_params": x_params,
            "s_hat": s_hat,
        }

        return out

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        vae_params = list(self.vae.parameters()) + list(self.noise_model.parameters())
        vae_optimizer = optim.Adamax(vae_params)
        optimizers.append(vae_optimizer)
        vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, factor=0.1, patience=50, verbose=True
        )
        schedulers.append(vae_scheduler)

        sd_params = self.s_decoder.parameters()
        sd_optimizer = optim.Adamax(sd_params)
        optimizers.append(sd_optimizer)
        sd_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            sd_optimizer, factor=0.1, patience=50, verbose=True
        )
        schedulers.append(sd_scheduler)

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        self.vae.mode_pred = False
        out = self(batch)

        x = (batch - self.data_mean) / self.data_std

        kl_loss = out["kl_loss"]
        reconstruction_loss = -self.noise_model.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_loss + reconstruction_loss
        self.manual_backward(elbo)
        self.log("train/elbo", elbo)
        self.log("train/kl_loss", kl_loss)
        self.log("train/reconstruction_loss", reconstruction_loss)

        sd_loss = self.s_decoder.loss(x, out["s_hat"]).mean()
        self.manual_backward(sd_loss)
        self.log("train/sd_loss", sd_loss)

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
        reconstruction_loss = -self.noise_model.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_loss + reconstruction_loss
        self.log("val/elbo", elbo)
        self.log("val/kl_loss", kl_loss)
        self.log("val/reconstruction_loss", reconstruction_loss)

        sd_loss = self.s_decoder.loss(x, out["s_hat"]).mean()
        self.log("val/sd_loss", sd_loss)

        if batch_idx == 0:
            idx = random.randint(0, batch.shape[0] - 1)
            out = self.forward(batch[idx : idx + 1].repeat_interleave(10, 0))
            mmse = torch.mean(out["s_hat"], 0, keepdim=True)
            self.log_tensorboard_images(batch[idx], "inputs/noisy")
            self.log_tensorboard_images(out["s_hat"][0], "outputs/sample 1")
            self.log_tensorboard_images(out["s_hat"][1], "outputs/sample 2")
            self.log_tensorboard_images(mmse[0], "outputs/mmse (10 samples)")

    def predict_step(self, batch, _):
        self.vae.mode_pred = True
        x = (batch - self.data_mean) / self.data_std

        vae_out = self.vae(x)
        s_code = vae_out["s_code"]
        s_hat = self.s_decoder(s_code)

        s_hat = s_hat * self.data_std + self.data_mean
        return s_hat
