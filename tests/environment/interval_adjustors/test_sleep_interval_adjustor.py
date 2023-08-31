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
    def test__init__(self, interval, offset):
        mod = cls(interval, offset)
        assert mod.interval == interval
        assert mod.offset == offset

        mod = cls(interval)  # デフォルト引数のテスト
        assert mod.offset == 0.0

    @pytest.mark.parametrize("interval, offset", params)
    def test_reset(self, interval, offset, mocker):
        mod = cls(interval, offset)
        with mocker.patch("time.perf_counter", return_value=0.0):
            assert mod.reset() == pytest.approx(time.perf_counter(), abs=eps)

    @pytest.mark.parametrize("interval, offset", params)
    def test_adjust(self, interval, offset):
        mod = cls(interval, offset)
        s = mod.reset()
        time_passed = mod.adjust()
        e = time.perf_counter()
        e_passed = mod._start_time
        assert e - s == pytest.approx(time_passed, abs=eps)  # 経過時間 == モデルが想定する経過時間
        assert e - s == pytest.approx(interval - offset, abs=eps)  # 経過時間 == 経過時間の理想値
        assert e_passed - s == pytest.approx(interval - offset, abs=eps)
