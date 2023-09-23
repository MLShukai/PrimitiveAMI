import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor

from .policy.stochastic_policy import StochasticPolicy
from .value.value import Value


class PolicyValueCommonNet(nn.Module):
    def __init__(
        self, base_model: nn.Module, policy: StochasticPolicy, value: Value, common_out_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.policy = policy
        self.value = value
        self.base_model = base_model
        self.fc1 = nn.Linear(common_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> tuple[distributions.Distribution, Tensor]:
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.policy(x), self.value(x)
