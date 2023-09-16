import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import TensorDataset

from src.data_collectors.data_collector import DataCollector
from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.data_collectors.sequential_data_collector import SequentialDataCollector as cls
from src.utils.step_record import RecordKeys as RK


class DataCollectorImple(DataCollector):
    def collect(self, step_record):
        return None

    def get_data(self):
        return None


class TestSequentialDataCollector:
    num_data_collectors = 3

    @pytest.fixture
    def data_collectors(self, mocker: MockerFixture):
        data_collectors = []
        for _ in range(self.num_data_collectors):
            mock = mocker.Mock(spec=DataCollectorImple)
            data_collectors.append(mock)
        return data_collectors

    def test__init__(self, data_collectors):
        mod = cls(data_collectors)
        assert mod.data_collectors is data_collectors

    def test_collect(self, data_collectors):
        dummy_record = {
            "": torch.empty(
                1,
            )
        }
        mod = cls(data_collectors)
        mod.collect(dummy_record)
        # 各データコレクタの`collect`が一度呼び出されていることを確認
        for data_collector in mod.data_collectors:
            assert data_collector.collect.call_count == 1

    def test_get_data(self, data_collectors):
        mod = cls(data_collectors)
        mod.get_data()
        for i, data_collector in enumerate(mod.data_collectors):
            assert data_collector.get_data.call_count == 1
