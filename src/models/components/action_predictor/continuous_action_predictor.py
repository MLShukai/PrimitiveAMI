from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .action_predictor import ActionPredictor


class ContinuousActionPredictor(ActionPredictor):
    def __init__(self, dim_embed: int, dim_action: int, dim_hidden: Optional[int] = None):
        """

        Args:
            dim_embed (int): Number of dimensions of embedded observation.
            dim_action (int): Number of dimensions of action.
            dim_hidden (int, optional): Number of dimensions of tensors in hidden layer. Defaults to None.
                                        See https://github.com/openai/large-scale-curiosity/blob/master/run.py#L51
                                        and https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L66
        """
        if dim_hidden is None:
            dim_hidden = dim_embed  # 2 * dim_embed -> dim_embed
        super().__init__()
        # 要調整
        self.fc_in = nn.Linear(dim_embed * 2, dim_hidden)
        self.fc_1 = nn.Linear(dim_hidden, dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, dim_action * 2)
        self.relu = (
            nn.ReLU()
        )  # https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L66 and https://github.com/openai/large-scale-curiosity/blob/master/utils.py#L115

    def forward(self, embed: Tensor, next_embed: Tensor) -> Tensor:
        x = torch.concat([embed, next_embed], dim=-1)
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        prev_action_hat, action_hat = torch.chunk(x, 2, dim=-1)
        return prev_action_hat, action_hat
