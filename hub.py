import torch
import pytorch_lightning as pl
from torch import optim
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_to_image


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
            "q_list": vae_out["q_list"],
            "p_list": vae_out["p_list"],
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
    
    def loss(self, batch, out):
        x = (batch - self.data_mean) / self.data_std

        losses = {}
        kl_div = self.vae.kl_divergence(out["q_list"], out["p_list"])
        kl_div = kl_div / x.numel()
        nll = -self.ar_decoder.loglikelihood(
            x, out["x_params"]
        ).mean()
        elbo = kl_div + nll
        losses["elbo"] = elbo
        losses["kl_div"] = kl_div
        losses["nll"] = nll

        sd_loss = self.s_decoder.loss(x, out["s_hat"]).mean()
        losses["sd_loss"] = sd_loss

        if self.direct_denoiser is not None:
            dd_loss = self.direct_denoiser.loss(
                out["s_hat"].detach(), out["s_direct"]
            ).mean()
            losses["dd_loss"] = dd_loss

        return losses

    def training_step(self, batch, batch_idx):
        out = self(batch)
        losses = self.loss(batch, out)

        self.manual_backward(losses["elbo"])
        self.log("train/elbo", losses["elbo"])
        self.log("train/kl_div", losses["kl_div"])
        self.log("train/nll", losses["nll"])

        self.manual_backward(losses["sd_loss"])
        self.log("train/sd_loss", losses["sd_loss"])

        if out["s_direct"] is not None:
            self.manual_backward(losses["dd_loss"])
            self.log("train/dd_loss", losses["dd_loss"])

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

    def log_image(self, img, img_name):
        normalised_img = (img - np.percentile(img, 1)) / (
            np.percentile(img, 99) - np.percentile(img, 1)
        )
        clamped_img = np.clip(normalised_img, 0, 1)
        self.trainer.logger.experiment.add_image(
            img_name, clamped_img, self.current_epoch
        )

    def log_val_images(self, batch, samples, mmse, direct):
        d = batch.ndim - 2
        if d == 1:
            for i in range(batch.size(1)):
                figure = plt.figure()
                plt.plot(batch[0, i].cpu().half(), label="Noisy", color="blue")
                for j in range(samples.size(0)):
                    plt.plot(samples[j][i].cpu().half(), color="orange", alpha=0.5)
                plt.plot(mmse[0, i].cpu().half(), label="Denoised", color="orange")
                if direct is not None:
                    plt.plot(direct[0, i].cpu().half(), label="Direct", color="green")
                plt.legend()
                img = plot_to_image(figure)
                self.log_image(img, f"channel_{i}")
                plt.close(figure)
        else:
            if d == 3:
                batch = batch[:, :, 0]
                samples = samples[:, :, 0]
                mmse = mmse[:, :, 0]
                if direct is not None:
                    direct = direct[:, :, 0]    
            self.log_image(batch[0].cpu().half().numpy(), "inputs/noisy")
            self.log_image(samples[0].cpu().half().numpy(), "outputs/sample 1")
            self.log_image(samples[1].cpu().half().numpy(), "outputs/sample 2")
            self.log_image(mmse[0].cpu().half().numpy(), "outputs/mmse (10 samples)")
            if direct is not None:
                self.log_image(direct[0].cpu().half().numpy(), "outputs/direct estimate")

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        losses = self.loss(batch, out)

        self.log("val/elbo", losses["elbo"])
        self.log("val/kl_div", losses["kl_div"])
        self.log("val/nll", losses["nll"])
        self.log("val/sd_loss", losses["sd_loss"])

        if out["s_direct"] is not None:
            self.log("val/dd_loss", losses["dd_loss"])

        if batch_idx == 0:
            idx = random.randint(0, batch.shape[0] - 1)
            out = self.forward(batch[idx : idx + 1].repeat_interleave(10, 0))
            mmse = torch.mean(out["s_hat"], 0, keepdim=True)
            x = (batch[idx : idx + 1] - self.data_mean) / self.data_std
            self.log_val_images(x, out["s_hat"], mmse, out["s_direct"])

    def predict_step(self, batch, _):
        self.eval()
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
        x = (x - self.data_mean) / self.data_std
        s_code = self.vae(x)["s_code"]
        s_hat = self.s_decoder(s_code)
        x_hat = self.ar_decoder.sample(s_code)
        s_hat = s_hat * self.data_std + self.data_mean
        x_hat = x_hat * self.data_std + self.data_mean

        return {"x_hat": x_hat, "s_hat": s_hat}
