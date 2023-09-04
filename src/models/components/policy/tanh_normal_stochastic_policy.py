import torch
import torch.distributions as distributions
import torch.nn as nn
import torchrl
from torch import Tensor
from torch.distributions.distribution import Distribution
from torchrl.modules import TanhNormal as _TanhNormal

from .stochastic_policy import StochasticPolicy


class TanhNormal(_TanhNormal):
    def entropy(self):
        return self.base_dist.base_dist.entropy()


class TanhNormalStochasticPolicy(StochasticPolicy):
    def __init__(self, dim_input: int, dim_out: int):
        super().__init__()
        self.fc_mean = nn.Linear(dim_input, dim_out)
        self.fc_std = nn.Linear(dim_input, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, input: Tensor) -> TanhNormal:
        mean = self.fc_mean(input)
        std = self.softplus(self.fc_std(input)) + 1e-7  # std 0 causes error
        tanh_norm_dist = TanhNormal(mean, std, upscale=1.0)
        return tanh_norm_dist
