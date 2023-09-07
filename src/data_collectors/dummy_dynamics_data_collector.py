from typing import Any

import torch
from torch import Tensor

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DummyDynamicsDataCollector(DataCollector):
    def __init__(self, action_dim, observation_dim):
        self.batch = 0
        self.action_dim = action_dim
        self.observation_dim = observation_dim

    def collect(self, step_record: dict[str, Any]):
        self.batch += 1

    def get_data(self) -> (Tensor, Tensor, Tensor, Tensor):
        prev_actions = torch.randn(self.batch, self.action_dim)
        observations = torch.randn(self.batch, self.observation_dim)
        actions = torch.randn(self.batch, self.action_dim)
        next_observations = torch.randn(self.batch, self.observation_dim)
        self.batch = 0
        return prev_actions, observations, actions, next_observations
