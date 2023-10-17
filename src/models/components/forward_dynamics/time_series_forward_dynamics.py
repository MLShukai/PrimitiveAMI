from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor


class TimeSeriesForwardDynamics(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:  # next_embed
        pass

    def get_hidden(self) -> Tensor:
        raise NotImplementedError

    def set_hidden(self, Tensor):
        raise NotImplementedError

    def reset_hidden(self):
        raise NotImplementedError
