import time

from .interval_adjustor import IntervalAdjustor


class BusyLoopIntervalAdjustor(IntervalAdjustor):
    def adjust(self) -> float:
        end_time = self._start_time + self._time_to_sleep

        while end_time > time.perf_counter():
            pass
        delta_time = time.perf_counter() - self._start_time
        self.reset()
        return delta_time
