from abc import ABCMeta, abstractmethod
from typing import Any


class Actuator(metaclass=ABCMeta):
    @abstractmethod
    def operate(action: Any) -> None:
        raise NotImplementedError
