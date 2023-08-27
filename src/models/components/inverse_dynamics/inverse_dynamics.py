import torch
import torch.nn as nn
from torch import Tensor

from ..action_predictor.action_predictor import ActionPredictor
from ..observation_encoder.observation_encoder import ObservationEncoder


class InverseDynamics(nn.Module):
    def __init__(
        self,
        action_predictor: ActionPredictor,
        observation_encoder: ObservationEncoder,
    ):
        super().__init__()
        self.action_predictor = action_predictor
        self.observation_encoder = observation_encoder

    def forward(self, obs: Tensor, next_obs: Tensor):
        embed = self.observation_encoder(obs)
        next_embed = self.observation_encoder(next_obs)
        prev_action_hat, action_hat = self.action_predictor(embed, next_embed)
        return prev_action_hat, action_hat
