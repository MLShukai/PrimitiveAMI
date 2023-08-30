from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution


class StochasticPolicy(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, input: Tensor) -> Distribution:
        pass
