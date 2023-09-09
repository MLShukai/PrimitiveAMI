from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import Adam, Optimizer

from .inverse_dynamics import InverseDynamics


class InverseDynamicsLitModule(LightningModule):
    def __init__(self, inverse_dynamics_net: InverseDynamics, optimizer: Optimizer):
        super().__init__()
        self.save_hyperparameters(
            logger=False
        )  # https://lightning.ai/docs/pytorch/1.6.2/common/hyperparameters.html#save-hyperparameters

    def configure_optimizers(self) -> Optimizer:
        return self.hparams.optimizer

    def training_step(self, batch: Tensor, batch_idx: int):
        prev_action, obs, action, next_obs = batch
        prev_action_hat, action_hat = self.hparams.inverse_dynamics_net(obs, next_obs)
        loss = mse_loss(prev_action_hat, prev_action) + mse_loss(action_hat, action)
        return loss
