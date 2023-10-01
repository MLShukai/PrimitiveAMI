from functools import partial

from lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

from .components.inverse_dynamics.inverse_dynamics import InverseDynamics


class InverseDynamicsLitModule(LightningModule):
    def __init__(self, inverse_dynamics_net: InverseDynamics, optimizer: partial[Optimizer]):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=["inverse_dynamics_net"]
        )  # https://lightning.ai/docs/pytorch/1.6.2/common/hyperparameters.html#save-hyperparameters
        self.net = inverse_dynamics_net

    def configure_optimizers(self) -> Optimizer:
        return self.hparams.optimizer(params=self.parameters())

    def training_step(self, batch: Tensor, batch_idx: int):
        prev_action, obs, action, next_obs = batch
        prev_action_hat, action_hat = self.net(obs, next_obs)
        loss = mse_loss(prev_action_hat, prev_action) + mse_loss(action_hat, action)

        self.log("inverse_dynamics/loss", loss, logger=True, on_step=True)

        return loss

    def forward(self, obs: Tensor, next_obs: Tensor):
        return self.net(obs, next_obs)
