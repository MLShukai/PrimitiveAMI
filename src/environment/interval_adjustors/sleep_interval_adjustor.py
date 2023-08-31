from .interval_adjustor import IntervalAdjustor
import time


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
        time.sleep(time_to_sleep)
        return time_to_sleep
