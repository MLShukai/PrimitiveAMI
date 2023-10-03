from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class DataCollector(ABC):
    """Abstract class for data collection."""

    @abstractmethod
    def collect(self, step_record: dict[str, Any]) -> None:
        """Collect data from agent's step record.

        Args:
            step_record (dict[str, Any]): A dictionary containing the data sucha as observations, actions, rewards, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Dataset:
        """Get collected data.

        Returns:
            Dataset: Collected data.
        """
        raise NotImplementedError
