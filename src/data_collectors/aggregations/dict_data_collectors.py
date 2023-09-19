from collections import UserDict
from typing import Any

from ..data_collector import DataCollector


class DictDataCollectors(UserDict, DataCollector):
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
        data = {}
        for key, value in self.items():
            data[key] = value.get_data()

        return data
