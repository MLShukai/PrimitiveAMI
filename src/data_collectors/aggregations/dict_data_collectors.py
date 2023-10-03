import warnings
from collections import UserDict
from pathlib import Path
from typing import Any

from ..data_collector import DataCollector
from .data_collectors_aggregation import DataCollectorsAggregation


class DictDataCollectors(UserDict, DataCollectorsAggregation):
    """MixIn Class for aggregating DataCollector objects like dict."""

    def __init__(self, **data_collectors: DataCollector) -> None:
        """Construct this class.

        Args:
            data_collectors (dict[str, DataCollector]): key-value pair of constructed data collector.
        """

        super().__init__(data_collectors)

    def collect(self, step_record: dict[str, Any]) -> None:
        """Call `collect` of all child data collector."""
        for collector in self.values():
            collector.collect(step_record)

    def get_data(self) -> dict[str, Any]:
        warnings.warn("The `get_data` method of Aggregation class is deprecated!", DeprecationWarning)
        data = {}
        for key, value in self.items():
            data[key] = value.get_data()

        return data

    def save_state_dict_to_files(self, dir_path: Path) -> None:
        """Save child data state to files."""
        for key, value in self.items():
            value.save_to_file(dir_path / f"{key}.pkl")
