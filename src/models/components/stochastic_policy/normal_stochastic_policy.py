import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from .stochastic_policy import StochasticPolicy


class NormalStochasticPolicy(StochasticPolicy):
    def __init__(self, dim_input: int, dim_out: int):
        super().__init__()
        self.fc_mean = nn.Linear(dim_input, dim_out)
        self.fc_std = nn.Linear(dim_input, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, input: Tensor) -> Distribution:
        mean = self.fc_mean(input)
        std = self.softplus(self.fc_std(input)) + 1e-7  # std 0 causes error
        norm_dist = distributions.normal.Normal(mean, std)
        return norm_dist
