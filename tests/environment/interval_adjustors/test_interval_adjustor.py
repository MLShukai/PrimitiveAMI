import time

import pytest

from src.environment.interval_adjustors.interval_adjustor import IntervalAdjustor

interval = 0.5
eps = 1e-3  # 待機時間の許容絶対誤差

class IntervalAdjustorImple(IntervalAdjustor):
    def __init__(self, interval, offset):
        super().__init__(interval, offset)
    
    def adjust():
        return

cls = IntervalAdjustorImple

class TestIntervalAdjustor:
    @pytest.mark.parametrize("interval, offset", [[1/10, 0.01]])
    def test__init__(self, interval, offset):
        mod = cls(interval, offset)
        assert mod.interval == interval
        assert mod.offset == offset

    def test_reset(self, mocker):
        interval, offset = 1/10, 0.0
        mod = cls(interval, offset)
        with mocker.patch("time.perf_counter", return_value=0.0) as mock:
            mod.reset()
            assert mod._start_time == time.perf_counter()

