import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.action_predictor.action_predictor import ActionPredictor


class ContinuousActionPredictor(ActionPredictor):
    def __init__(self, dim_embed: int, dim_action: int):
        super().__init__()
        # 要調整
        self.fc1 = nn.Linear(dim_embed * 2, dim_embed * 2 + dim_action)
        self.fc2 = nn.Linear(dim_embed * 2 + dim_action, dim_action)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()

    def forward(self, embed: Tensor, next_embed: Tensor, action: Tensor) -> Tensor:
        x = torch.concat([embed, next_embed], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        action_hat = self.fc2(x)
        loss = self.mse_loss(action_hat, action)
        return loss, action_hat
