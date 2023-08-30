import math
from abc import ABCMeta, abstractmethod
from time import perf_counter, sleep


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


class SleepIntervalAdjustor(IntervalAdjustor):
    def __init__(self, interval: float) -> None:
        super().__init__(interval)

    def adjust(self, offset: float = 0.0) -> float:
        """Adjust time by time.sleep.

        Args:
            offset (float, optional): Offset value not to delay processing. Defaults to 0.0.

        Returns:
            float: Ideal value of time waited.
        """
        time_to_sleep = self.interval - offset
        sleep(time_to_sleep)
        return time_to_sleep


class BusyLoopIntervalAdjustor(IntervalAdjustor):
    def __init__(self, interval: float) -> None:
        super().__init__(interval)

    def adjust(self, offset: float = 0.0) -> float:
        time_to_sleep = self.interval - offset
        start_time = self.reset()
        while start_time + time_to_sleep >= perf_counter():
            pass
        return time_to_sleep
