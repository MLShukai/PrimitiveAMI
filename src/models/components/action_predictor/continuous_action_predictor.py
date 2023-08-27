import torch
import torch.nn as nn
from torch import Tensor

from .action_predictor import ActionPredictor


class ContinuousActionPredictor(ActionPredictor):
    def __init__(self, dim_embed: int, dim_action: int):
        super().__init__()
        # 要調整
        self.fc_in = nn.Linear(dim_embed * 2, dim_embed * 2 + dim_action * 2)
        self.fc_1 = nn.Linear(dim_embed * 2 + dim_action * 2, dim_embed * 2 + dim_action * 2)
        self.fc_out = nn.Linear(dim_embed * 2 + dim_action * 2, dim_action * 2)
        self.relu = nn.ReLU()

    def forward(self, embed: Tensor, next_embed: Tensor) -> Tensor:
        x = torch.concat([embed, next_embed], dim=-1)
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        prev_action_hat, action_hat = torch.chunk(x, 2, dim=-1)
        return prev_action_hat, action_hat
