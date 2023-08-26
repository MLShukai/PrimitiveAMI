from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor


class ObservationEncoder(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, obs: Tensor) -> Tensor:  # (embed)
        pass
