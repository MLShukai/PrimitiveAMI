import torch
import torch.nn as nn
from torch import Tensor

from .forward_dynamics import ForwardDynamics


class DenseNetForwardDynamics(ForwardDynamics):
    def __init__(self, dim_action: int, dim_embed: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_action + dim_embed + dim_action, dim_embed * 2)
        self.fc2 = nn.Linear(dim_embed * 2, dim_embed * 2)
        self.fc3 = nn.Linear(dim_embed * 2, dim_embed)
        self.relu = nn.ReLU()

    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:
        x = torch.concat([prev_action, embed, action], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        next_embed = self.fc3(x)
        return next_embed
