from typing import Any

import torch
from torch import Tensor

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DummyDynamicsDataCollector(DataCollector):
    def __init__(self, action_dim, observation_dim):
        self.action_dim = action_dim
        self.observation_dim = observation_dim

    def collect(self, step_record: dict[str, Any]):
        pass

    def get_data(self) -> (Tensor, Tensor, Tensor, Tensor):
        batch = 3
        prev_actions = torch.randn(batch, self.action_dim)
        observations = torch.randn(batch, self.observation_dim)
        actions = torch.randn(batch, self.action_dim)
        next_observations = torch.randn(batch, self.observation_dim)
        return prev_actions, observations, actions, next_observations
