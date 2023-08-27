from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor


class ActionPredictor(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, embed: Tensor, next_embed: Tensor) -> (Tensor, Tensor):  # (prev_action_hat, action_hat)
        pass
