import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor

from .policy.stochastic_policy import StochasticPolicy
from .value.value import Value


class PolicyValueCommonNet(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        policy: StochasticPolicy,
        value: Value,
    ):
        super().__init__()
        self.policy = policy
        self.value = value
        self.base_model = base_model

    def forward(self, x: Tensor) -> tuple[distributions.Distribution, Tensor]:
        x = self.base_model(x)
        return self.policy(x), self.value(x)
