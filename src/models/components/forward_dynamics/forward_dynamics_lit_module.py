from functools import partial

import torch
from lightning import LightningModule
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

from ..observation_encoder.observation_encoder import ObservationEncoder
from .forward_dynamics import ForwardDynamics


class ForwardDynamicsLitModule(LightningModule):
    def __init__(
        self, obs_encoder: ObservationEncoder, forward_dynamics_net: ForwardDynamics, optimizer: partial[Optimizer]
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.forward_dynamics_net = forward_dynamics_net
        self.optimizer = optimizer

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.forward_dynamics_net.parameters())

    def training_step(self, batch, batch_idx):
        prev_action, obs, action, next_obs = batch
        with torch.no_grad():
            embed_obs = self.obs_encoder(obs)
            embed_next_obs = self.obs_encoder(next_obs)
        embed_next_obs_hat = self.forward_dynamics_net(prev_action, embed_obs, action)
        loss = mse_loss(embed_next_obs_hat, embed_next_obs)
        return loss
