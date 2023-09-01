import time

from .interval_adjustor import IntervalAdjustor


class SleepIntervalAdjustor(IntervalAdjustor):
    def adjust(self) -> float:
        """Adjust time by time.sleep.

        Args:
            offset (float, optional): Offset value not to delay processing. Defaults to 0.0.

        Returns:
            float: Ideal value of time waited.
        """
        if (remaining_time := (self._start_time + self._time_to_sleep) - time.perf_counter()) > 0:
            time.sleep(remaining_time)
        delta_time = time.perf_counter() - self._start_time
        self.reset()
        return delta_time
