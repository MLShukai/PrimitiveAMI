from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class ObsDataCollector(DataCollector):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.observations = []

    def collect(self, step_record: dict[str, Tensor]):
        obs = step_record[RK.OBSERVATION]
        if len(self.observations) < self.max_size:
            self.observations.append(obs)
        else:
            rand_ind = np.random.randint(0, self.max_size)
            self.observations[rand_ind] = obs

    def get_data(self) -> TensorDataset:
        observations = torch.stack(self.observations)
        return TensorDataset(observations)

    def state_dict(self) -> dict[str, Any]:
        return {"observations": self.observations}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.observations = state_dict["observations"][: self.max_size]
