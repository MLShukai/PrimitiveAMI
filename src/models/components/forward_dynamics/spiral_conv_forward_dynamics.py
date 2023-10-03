import torch
import torch.nn as nn
from torch import Tensor

from ..spiral_conv import Architecture
from .time_series_forward_dynamics import TimeSeriesForwardDynamics


class SpiralConvForwardDynamics(TimeSeriesForwardDynamics):
    def __init__(self, dim_action: int, dim_embed: int, depth: int, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.spiral_conv = Architecture(depth, dim, dim_ff_scale, dropout)
        self.fc_in = nn.Linear(dim_embed + dim_action, dim)
        self.fc_out = nn.Linear(dim, dim_embed)

    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:
        x = torch.concat([embed, action], dim=-1).unsqueeze(0)
        x = self.fc_in(x)
        x = self.spiral_conv(x)
        x = self.fc_out(x)
        return x.squeeze(0)

    def get_hidden(self) -> [(Tensor, Tensor)]:
        return self.spiral_conv.get_hidden()

    def set_hidden(self, hidden_list):
        self.spiral_conv.set_hidden(hidden_list)
