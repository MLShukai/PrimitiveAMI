from .data_collector import DataCollector
from torch import Tensor
from ..utils.step_record import RecordKeys as RK
import numpy as np
from torch.utils.data import TensorDataset
import torch


class VAEDataCollector(DataCollector):
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
