import time

from .interval_adjustor import IntervalAdjustor


class BusyLoopIntervalAdjustor(IntervalAdjustor):
    def __init__(self, interval: float, offset: float = 0.0) -> None:
        super().__init__(interval, offset)

    def adjust(self) -> float:
        interval, offset = self.interval, self.offset
        time_to_sleep = interval - offset
        while self._start_time + time_to_sleep > time.perf_counter():
            pass
        self.reset()
        return time_to_sleep
