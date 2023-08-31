from abc import ABCMeta, abstractmethod
from time import perf_counter


import math
class IntervalAdjustor(metaclass=ABCMeta):
    _start_time: float = -math.inf
    def __init__(self, interval: float) -> None:
        self.reset()
        self.interval = interval

    def reset(self) -> float:
        """Reset start time of this adjustor.

        Returns:
            float: Start time after resetting timer.
        """
        self._start_time = perf_counter()
        return self._start_time

    @abstractmethod
    def adjust(self) -> float:
        raise NotImplementedError
