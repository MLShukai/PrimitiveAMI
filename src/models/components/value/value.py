from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor


class Value(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, input: Tensor):
        pass
