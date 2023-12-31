from functools import partial

import torch
from lightning import LightningModule
from torch import Tensor
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import kl_div, mse_loss
from torch.optim import Optimizer

from .components.observation_encoder.vae import VAE


class VAELitModule(LightningModule):
    def __init__(self, vae_net: VAE, optimizer: partial[Optimizer], kl_coef: float = 1.0):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["vae_net"])
        self.net = vae_net

    def configure_optimizers(self) -> Optimizer:
        return self.hparams.optimizer(params=self.parameters())

    def training_step(self, batch: list[Tensor], batch_idx: int):
        (x,) = batch
        x_reconstructed, z_dist = self.net(x)
        rec_loss = mse_loss(x, x_reconstructed)
        kl_loss = kl_divergence(z_dist, Normal(torch.zeros_like(z_dist.mean), torch.ones_like(z_dist.stddev))).mean()
        self.log("vae/kl_loss", kl_loss, prog_bar=True)
        self.log("vae/reconstruction_loss", rec_loss, prog_bar=True)
        return rec_loss + self.hparams.kl_coef * kl_loss

    def forward(self, x: Tensor):
        return self.net(x)
