import warnings
from pathlib import Path
from typing import Any

from ..data_collector import DataCollector
from .data_collectors_aggregation import DataCollectorsAggregation


class SequentialDataCollectors(DataCollectorsAggregation):
    def __init__(self, *data_collectors: DataCollector) -> None:
        self.data_collectors = data_collectors

    def collect(self, step_record: dict[str, Any]) -> None:
        for collector in self.data_collectors:
            collector.collect(step_record)

    def get_data(self) -> list[Any]:
        warnings.warn("The `get_data` method of Aggregation class is deprecated!", DeprecationWarning)
        data = [collector.get_data() for collector in self.data_collectors]
        return data

    def save_state_dict_to_files(self, dir_path: Path) -> None:
        for idx, collector in enumerate(self.data_collectors):
            dist_path = dir_path / f"{collector.__class__.__name__}.{idx}.pkl"
            collector.save_to_file(dist_path)
