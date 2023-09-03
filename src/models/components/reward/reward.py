from abc import ABCMeta, abstractmethod
from typing import Any

import torch.nn as nn
from torch import Tensor


class Reward(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def reward(self, *args: Any, **kwds: Any) -> Tensor:  # reward
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:  # reward
        return self.reward(*args, **kwds)
