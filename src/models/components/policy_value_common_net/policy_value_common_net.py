import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor

from ..policy.stochastic_policy import StochasticPolicy
from ..value.value import Value


class PolicyValueCommonNet(nn.Module):
    def __init__(self, policy: StochasticPolicy, value: Value, height: int, width: int, dim_hidden: int):
        super().__init__()
        self.policy = policy
        self.value = value
        self.conv2d1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(
            ((((height - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * ((((width - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * 64,
            dim_hidden,
        )

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return self.policy(x), self.value(x)
