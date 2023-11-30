import torch
import torch.nn as nn
from torch import Tensor

from ....utils.model import MultiEmbeddings
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

    def get_hidden(self) -> list[tuple[Tensor, Tensor]]:
        return self.spiral_conv.get_hidden()

    def set_hidden(self, hidden_list):
        self.spiral_conv.set_hidden(hidden_list)

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()


class DiscreteActionSCFD(SpiralConvForwardDynamics):
    """Discrete action version of spiral conv forward dynamcis."""

    def __init__(
        self,
        action_choices_per_category: list[int],
        action_embedding_dim: int,
        dim_embed: int,
        depth: int,
        dim: int,
        dim_ff_scale: float,
        dropout: float,
    ):

        dim_action = len(action_choices_per_category) * action_embedding_dim
        super().__init__(dim_action, dim_embed, depth, dim, dim_ff_scale, dropout)
        self.embedding = MultiEmbeddings(action_choices_per_category, action_embedding_dim)

    def forward(self, prev_action: Tensor, embed: Tensor, action: Tensor) -> Tensor:
        """Compute forward path.

        Shape:
            prev_action: (*, num_action_types)
            embed: (*, dim_embed),
            action: (*, num_action_types)
        """

        prev_action = self.embedding(prev_action).flatten(-2)
        action = self.embedding(action).flatten(-2)

        return super().forward(prev_action, embed, action)
