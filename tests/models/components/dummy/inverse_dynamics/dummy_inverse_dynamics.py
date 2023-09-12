import torch
import torch.nn as nn


class DummyInverseDynamics(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(2 * in_dim, 2 * out_dim)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor):
        x = torch.cat((obs, next_obs), dim=-1)
        return torch.chunk(self.layer(x), 2, dim=-1)
