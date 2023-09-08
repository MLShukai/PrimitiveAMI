from typing import Any

import torch
from torch import Tensor

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DummyDynamicsDataCollector(DataCollector):
    def __init__(self, action_dim: int, observation_dim: int, get_size: int):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.get_size = get_size

    def collect(self, step_record: dict[str, Any]):
        pass

    def get_data(self) -> (Tensor, Tensor, Tensor, Tensor):
        prev_actions = torch.randn(self.get_size, self.action_dim)
        observations = torch.randn(self.get_size, self.observation_dim)
        actions = torch.randn(self.get_size, self.action_dim)
        next_observations = torch.randn(self.get_size, self.observation_dim)
        return prev_actions, observations, actions, next_observations
