import torch.nn as nn
from torch import Tensor

from .reward import Reward


class CuriosityReward(Reward):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction="none")

    def reward(self, obs_hat: Tensor, obs: Tensor) -> Tensor:
        return self.mseloss(obs_hat, obs).mean(dim=-1)
