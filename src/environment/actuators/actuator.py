from abc import ABCMeta, abstractmethod
from typing import Any


class Actuator(metaclass=ABCMeta):
    @abstractmethod
    def operate(action: Any) -> None:
        raise NotImplementedError

    def setup(self):
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self):
        """Called at the end of interaction with the agent."""
        pass
