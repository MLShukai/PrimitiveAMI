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


class ActuatorWrapper(Actuator):
    """Base wrapper class for actuator.

    You must override `wrap_action` method for wrapping action.
    """

    def __init__(self, actuator: Actuator) -> None:
        self._actuator = actuator

    @abstractmethod
    def wrap_action(self, action: Any) -> Any:
        raise NotImplementedError

    def operate(self, action: Any) -> None:
        self._actuator.operate(self.wrap_action(action))

    def setup(self) -> None:
        self._actuator.setup()

    def teardown(self) -> None:
        self._actuator.teardown()
