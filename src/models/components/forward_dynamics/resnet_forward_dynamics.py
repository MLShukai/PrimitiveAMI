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
        dim_hidden_cat_action = dim_action + dim_hidden + dim_action
        self.fc1_list = nn.ModuleList([nn.Linear(dim_hidden_cat_action, dim_hidden) for _ in range(depth)])
        self.fc2_list = nn.ModuleList([nn.Linear(dim_hidden_cat_action, dim_hidden) for _ in range(depth)])
        self.fc_out = nn.Linear(dim_hidden_cat_action, dim_embed)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:
        x = self.add_action(prev_action, embed, action)
        x = self.fc_in(x)
        for i in range(self.depth):
            x_ = self.fc1_list[i](self.add_action(prev_action, x, action))
            x_ = self.leaky_relu(x_)
            x_ = self.fc2_list[i](self.add_action(prev_action, x_, action))
            x = x_ + x
        next_embed = self.fc_out(self.add_action(prev_action, x, action))
        return next_embed

    @staticmethod
    def add_action(prev_action, x, action):
        return torch.concat([prev_action, x, action], dim=-1)
