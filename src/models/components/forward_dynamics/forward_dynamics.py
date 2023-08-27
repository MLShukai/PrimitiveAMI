from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor


class ForwardDynamics(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:  # next_embed
        pass
