from functools import partial
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.distributions import Distribution
from torchmetrics import MeanMetric

from .components.policy_value_common_net import PolicyValueCommonNet


class PPOLitModule(pl.LightningModule):
    """Proximal Policy Optimization Lightning Module.

    Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    """

    def __init__(
        self,
        net: PolicyValueCommonNet,
        optimizer: partial[torch.optim.Optimizer],
        norm_advantage: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        entropy_coef: float = 0.01,
        vfunc_coef: float = 0.5,
    ) -> None:
        """Initialize a PPOLitModule.

        Args:
            net (PolicyValueCommonNet): Model training by ppo method.
            optimizer (Optimizer): Partial instantiation of optimizer.
            norm_advantage (bool): Toggles advantages normalization.
            clip_coef (float): The surrogate clipping coefficient.
            clip_vloss (bool): Toggles whether or not to use a clipped loss for the value function, as per the paper.
            entropy_coef (float): Coefficient of the entropy.
            vfunc_coef: Coefficient of the value function.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # Averaging loss across batches
        self.loss = MeanMetric()
        self.value_loss = MeanMetric()
        self.policy_loss = MeanMetric()
        self.clipflac = MeanMetric()

    def forward(self, obs: Tensor) -> tuple[Distribution, Tensor]:
        return self.net(obs)

    def model_step(self, batch: tuple[Tensor, ...]) -> dict[str, Any]:
        """Perform a single model step on a batch of data.
        
        Shape:
            obses: (batch, channels, height, width)
            actions: (batch, action_size)
            logprobs: (batch, action_size)
            advantages: (batch,)
            returns: (batch,)
            values: (batch,)
        """
        # Setup
        obses, actions, logprobs, advantanges, returns, values = batch

        new_action_dist, new_values = self.forward(obses)
        new_logprobs = new_action_dist.log_prob(actions)
        entropy = new_action_dist.entropy()

        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.hparams.clip_coef).float().mean()

        if self.hparams.norm_advantage:
            advantanges = (advantanges - advantanges.mean()) / (advantanges.std() + 1e-8)

        if advantanges.ndim == 1:
            advantanges = advantanges.unsqueeze(1)

        # Policy loss
        pg_loss1 = -advantanges * ratio
        pg_loss2 = -advantanges * torch.clamp(ratio, 1 - self.hparams.clip_coef, 1 + self.hparams.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = new_values.flatten()
        if self.hparams.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(new_values - values, -self.hparams.clip_coef, self.hparams.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        loss = pg_loss - self.hparams.entropy_coef * entropy_loss + v_loss * self.hparams.vfunc_coef

        # Output
        output = {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": clipfracs,
        }

        return output

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        """Write training step."""

        output = self.model_step(batch)
        self.loss(output["loss"])
        self.value_loss(output["value_loss"])
        self.policy_loss(output["policy_loss"])
        self.clipflac(output["clipfrac"])

        prefix = "train/"
        for (name, value) in output.items():
            self.log(prefix + name, value)

        self.log(prefix + "mean_loss", self.loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(prefix + "value_loss", self.value_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(prefix + "policy_loss", self.policy_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(prefix + "clipfrac", self.clipflac, on_step=False, on_epoch=True, prog_bar=True)

        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.hparams.optimizer(params=self.parameters())
