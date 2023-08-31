from .interval_adjustor import IntervalAdjustor
import time


class BusyLoopIntervalAdjustor(IntervalAdjustor):
    def __init__(self, interval: float) -> None:
        super().__init__(interval)

    def adjust(self, offset: float = 0.0) -> float:
        time_to_sleep = self.interval - offset
        start_time = self.reset()
        while start_time + time_to_sleep >= time.perf_counter():
            pass
        return time_to_sleep
