from typing import Any

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DummyDynamicsDataCollector(DataCollector):
    def __init__(self, action_shape: tuple[int], observation_shape: tuple[int], get_size: int):
        self.action_shape = action_shape
        self.observation_shape = observation_shape
        self.get_size = get_size

    def collect(self, step_record: dict[str, Tensor]):
        pass

    def get_data(self) -> TensorDataset:
        prev_actions = torch.randn(self.get_size, *self.action_shape)
        observations = torch.randn(self.get_size, *self.observation_shape)
        actions = torch.randn(self.get_size, *self.action_shape)
        next_observations = torch.randn(self.get_size, *self.observation_shape)
        return TensorDataset(prev_actions, observations, actions, next_observations)
