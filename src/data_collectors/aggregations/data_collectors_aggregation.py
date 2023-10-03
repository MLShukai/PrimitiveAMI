from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DataCollectorsAggregation(ABC):
    """Abstract class for aggregating data collectors.

    Gather data collectors, and call all `collect` method of the
    gathered them with this class `collect` method.
    """

    @abstractmethod
    def collect(self, step_record: dict[str, Any]) -> None:
        """Call child `collect` method.

        Args:
            step_record (dict[str, Any]): A dictionary containing the data sucha as observations, actions, rewards, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def save_state_dict_to_files(self, dir_path: Path) -> None:
        """Create a save file name for each DataCollector held in this class
        and save them states.

        Args:
            dir_path (Path): Path to the saving directory.
        """
        raise NotImplementedError
