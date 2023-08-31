import torch
import torch.nn as nn
import torchrl
from torch import Tensor
from torch.distributions.distribution import Distribution

from .stochastic_policy import StochasticPolicy


class DeterministicStochasticPolicy(StochasticPolicy):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Distribution:
        delta_dist = torchrl.modules.Delta(input)
        return delta_dist
