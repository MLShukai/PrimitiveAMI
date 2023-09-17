from typing import Any

from .data_collector import DataCollector


class SequentialDataCollectors:
    def __init__(self, *data_collectors: DataCollector) -> None:
        self.data_collectors = data_collectors

    def collect(self, step_record: dict[str, Any]) -> None:
        for collector in self.data_collectors:
            collector.collect(step_record)

    def get_data(self) -> list[Any]:
        data = [collector.get_data() for collector in self.data_collectors]
        return data