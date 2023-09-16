from typing import Any

from torch.utils.data import TensorDataset

from .data_collector import DataCollector


class SequentialDataCollector:
    def __init__(self, data_collectors: list[DataCollector]) -> None:
        self.data_collectors = data_collectors

    def collect(self, step_record: dict[str, Any]) -> None:
        for collector in self.data_collectors:
            collector.collect(step_record)

    def get_data(self) -> list[TensorDataset]:
        data = [collector.get_data() for collector in self.data_collectors]
        return data
