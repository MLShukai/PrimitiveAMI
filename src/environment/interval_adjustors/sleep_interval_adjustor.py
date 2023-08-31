import time

from .interval_adjustor import IntervalAdjustor


class SleepIntervalAdjustor(IntervalAdjustor):
    def __init__(self, interval: float, offset: float = 0.0) -> None:
        super().__init__(interval, offset)

    def adjust(self) -> float:
        """Adjust time by time.sleep.

        Args:
            offset (float, optional): Offset value not to delay processing. Defaults to 0.0.

        Returns:
            float: Ideal value of time waited.
        """
        interval, offset = self.interval, self.offset
        time_to_sleep = interval - offset
        if (remaining_time := (self._start_time + time_to_sleep) - time.perf_counter()) > 0:
            time.sleep(remaining_time)
        self.reset()
        return time_to_sleep
