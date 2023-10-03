import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar

from torch.utils.data import Dataset

SelfDataCollector = TypeVar("SelfDataCollector", bound="DataCollector")


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

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return dictionary that contains the internal state of data
        collector.

        Returns:
            dict[str, Any]: The internal state of data collector.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from state dict.

        Args:
            state_dict (dict[str, Any]): The internal state of data collector.
        """
        raise NotImplementedError

    def save_to_file(self, dist_path: Path) -> None:
        """Save state dict to file with pickling.

        Args:
            dist_path (Path): Distination of state dict file.
        """
        with open(dist_path, mode="wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load_from_file(
        cls: Type[SelfDataCollector], state_dict_path: Path, *init_args: Any, **init_kwds: Any
    ) -> SelfDataCollector:
        """Instantiate and load from state dict file.

        Args:
            state_dict_path (Path): File path to the state dict file.
            *init_args (Any): constructing args.
            **init_kwds (Any): constructing keyword args.
        """

        data_collector = cls(*init_args, **init_kwds)
        with open(state_dict_path, mode="rb") as f:
            data_collector.load_state_dict(pickle.load(f))

        return data_collector
