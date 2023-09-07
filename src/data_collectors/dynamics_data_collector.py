from typing import Any

import torch
from torch import Tensor

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DynamicsDataCollector(DataCollector):
    def __init__(self):
        self.prev_actions = []
        self.observations = []
        self.actions = []
        self.next_observations = []

    def collect(self, step_record: dict[str, Any]):
        self.prev_actions.append(step_record[RK.PREVIOUS_ACTION])
        self.observations.append(step_record[RK.OBSERVATION])
        self.actions.append(step_record[RK.ACTION])
        self.next_observations.append(step_record[RK.NEXT_OBSERVATION])

    def get_data(self) -> (Tensor, Tensor, Tensor, Tensor):
        prev_actions = torch.stack(self.prev_actions)
        self.prev_actions = []
        observations = torch.stack(self.observations)
        self.observations = []
        actions = torch.stack(self.actions)
        self.actions = []
        next_observations = torch.stack(self.next_observations)
        self.next_observations = []
        return prev_actions, observations, actions, next_observations
