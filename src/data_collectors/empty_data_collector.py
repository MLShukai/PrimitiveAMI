from typing import Any

import torch
from torch.utils.data import TensorDataset

from .data_collector import DataCollector


class EmptyDataCollector(DataCollector):
    """Do nothing."""

    def collect(self, step_record: dict[str, Any]) -> None:
        pass

    def get_data(self) -> TensorDataset:
        return TensorDataset(torch.zeros(0))

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
