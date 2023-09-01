import time

import pytest

from src.environment.interval_adjustors.sleep_interval_adjustor import (
    SleepIntervalAdjustor as cls,
)

params = [
    (1 / 10, 0.0),
    (1 / 10, 0.01),
    (1 / 30, 0.01),
]
eps = 1e-3


class TestSleepIntervalAdjustor:
    @pytest.mark.parametrize("interval, offset", params)
    def test_adjust(self, interval, offset):
        mod = cls(interval, offset)
        s = mod.reset()
        time_passed = mod.adjust()
        e = time.perf_counter()
        e_passed = mod._start_time
        assert e - s == pytest.approx(time_passed, abs=eps)  # 経過時間 == モデルが想定する経過時間
        assert e - s == pytest.approx(mod._time_to_sleep, abs=eps)  # 経過時間 == 経過時間の理想値
        assert e_passed - s == pytest.approx(mod._time_to_sleep, abs=eps)
