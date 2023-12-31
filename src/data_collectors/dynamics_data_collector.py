from typing import Any

import numpy
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class DynamicsDataCollector(DataCollector):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.prev_actions = []
        self.observations = []
        self.actions = []
        self.next_observations = []

    def collect(self, step_record: dict[str, Tensor]):
        if len(self.prev_actions) < self.max_size:
            self.prev_actions.append(step_record[RK.PREVIOUS_ACTION].clone().cpu())
            self.observations.append(step_record[RK.OBSERVATION].clone().cpu())
            self.actions.append(step_record[RK.ACTION].clone().cpu())
            self.next_observations.append(step_record[RK.NEXT_OBSERVATION].clone().cpu())
        else:
            rand_ind = numpy.random.randint(0, self.max_size)
            self.prev_actions[rand_ind] = step_record[RK.PREVIOUS_ACTION].clone().cpu()
            self.observations[rand_ind] = step_record[RK.OBSERVATION].clone().cpu()
            self.actions[rand_ind] = step_record[RK.ACTION].clone().cpu()
            self.next_observations[rand_ind] = step_record[RK.NEXT_OBSERVATION].clone().cpu()

    def get_data(self) -> TensorDataset:
        prev_actions = torch.stack(self.prev_actions)
        observations = torch.stack(self.observations)
        actions = torch.stack(self.actions)
        next_observations = torch.stack(self.next_observations)
        return TensorDataset(prev_actions, observations, actions, next_observations)

    def state_dict(self) -> dict[str, Any]:
        state = {
            "prev_actions": self.prev_actions,
            "observations": self.observations,
            "actions": self.actions,
            "next_observations": self.next_observations,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:

        self.prev_actions = state_dict["prev_actions"][: self.max_size]
        self.observations = state_dict["observations"][: self.max_size]
        self.actions = state_dict["actions"][: self.max_size]
        self.next_observations = state_dict["next_observations"][: self.max_size]
