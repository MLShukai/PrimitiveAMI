from time import perf_counter

import pytest

from src.environment.interval_adjustor import (
    BusyLoopIntervalAdjustor,
    SleepIntervalAdjustor,
)

interval = 0.5
eps = 1e-3  # 待機時間の許容絶対誤差


class TestSleepIntervalAdjustor:
    @pytest.fixture
    def adjustor(self):
        return SleepIntervalAdjustor(interval)

    def test__init__(self, adjustor):
        assert type(adjustor._start_time) is float
        assert adjustor.interval == interval

    def test_reset(self, adjustor):
        assert adjustor.reset() == pytest.approx(perf_counter())

    @pytest.mark.parametrize("offset", (0.0, 0.1))
    def test_adjust(self, adjustor, offset):
        s = perf_counter()
        time_to_sleep = adjustor.adjust(offset)
        e = perf_counter()
        assert e - s == pytest.approx(time_to_sleep, abs=eps)


class TestBusyLoopIntervalAdjustor:
    """reset, __init__はSleepIntervalAdjustorと共通."""

    @pytest.fixture
    def adjustor(self):
        return BusyLoopIntervalAdjustor(interval)

    @pytest.mark.parametrize("offset", (0.0, 0.1))
    def test_adjust(self, adjustor, offset):
        s = perf_counter()
        time_to_sleep = adjustor.adjust(offset)
        e = perf_counter()
        assert e - s == pytest.approx(time_to_sleep, abs=eps)
