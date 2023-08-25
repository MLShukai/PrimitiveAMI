from action_predictor import ActionPredictor
import torch
import torch.nn as nn
from torch import Tensor

class DiscreteActionPredictor(ActionPredictor):
    def __init__(self, dim_embed: int, dim_action: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim_embed*2, dim_embed*2+dim_action)
        self.linear_2 = nn.Linear(dim_embed*2+dim_action, dim_action)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, embed: Tensor, next_embed: Tensor) -> Tensor:
        x = torch.concat([embed, next_embed], dim=-1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x

