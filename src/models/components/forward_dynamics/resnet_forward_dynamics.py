import torch
import torch.nn as nn
from torch import Tensor

from .forward_dynamics import ForwardDynamics


# https://github.com/openai/large-scale-curiosity/blob/e0a698676d19307a095cd4ac1991c4e4e70e56fb/dynamics.py#L43-L65
class ResNetForwardDynamics(ForwardDynamics):
    def __init__(self, dim_action: int, dim_embed: int, dim_hidden: int = 512, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.fc_in = nn.Linear(dim_action + dim_embed + dim_action, dim_hidden)
        self.fc1_list = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(depth)])
        self.fc2_list = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(depth)])
        self.fc_out = nn.Linear(dim_hidden, dim_embed)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:
        x = torch.concat([prev_action, embed, action], dim=-1)
        x = self.fc_in(x)
        for i in range(self.depth):
            x_ = self.fc1_list[i](x)
            x_ = self.leaky_relu(x_)
            x_ = self.fc2_list[i](x_)
            x = x_ + x
        next_embed = self.fc_out(x)
        return next_embed
