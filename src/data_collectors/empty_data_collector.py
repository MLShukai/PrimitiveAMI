from typing import Any

from .data_collector import DataCollector


class EmptyDataCollector(DataCollector):
    """Do nothing."""

    def collect(self, step_record: dict[str, Any]) -> None:
        pass

    def get_data(self) -> None:
        return None
