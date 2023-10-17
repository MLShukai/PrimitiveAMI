from functools import partial

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

from .components.forward_dynamics.time_series_forward_dynamics import (
    TimeSeriesForwardDynamics,
)
from .components.observation_encoder.observation_encoder import ObservationEncoder


class TimeSeriesForwardDynamicsLitModule(LightningModule):
    def __init__(
        self,
        obs_encoder: ObservationEncoder,
        forward_dynamics_net: TimeSeriesForwardDynamics,
        optimizer: partial[Optimizer],
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.forward_dynamics_net = forward_dynamics_net
        self.optimizer = optimizer
        self.current_hidden = None

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.forward_dynamics_net.parameters())

    def training_step(self, batch, batch_idx):
        prev_action, obs, action, next_obs = batch
        with torch.no_grad():
            embed_obs = self.obs_encoder(obs)
            embed_next_obs = self.obs_encoder(next_obs)

        embed_next_obs_hat = self.forward_dynamics_net(prev_action, embed_obs, action)
        loss = mse_loss(embed_next_obs_hat, embed_next_obs)

        self.log("forward_dynamics/loss", loss)

        return loss

    def on_train_start(self):
        self.current_hidden = self.forward_dynamics_net.get_hidden()

    def on_train_epoch_start(self):
        self.forward_dynamics_net.reset_hidden()

    def on_train_end(self):
        self.forward_dynamics_net.set_hidden(self.current_hidden)
