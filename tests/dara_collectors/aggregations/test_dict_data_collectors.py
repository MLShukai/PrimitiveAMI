from typing import Any

import pytest
from pytest_mock import MockerFixture

from src.data_collectors.aggregations.dict_data_collectors import DictDataCollectors
from src.data_collectors.data_collector import DataCollector


class TestDictDataCollectors:
    def get_mock_data_collector(self, mocker: MockerFixture, data: Any) -> DataCollector:
        collector = mocker.Mock(spec=DataCollector)
        collector.get_data.return_value = data
        return collector

    @pytest.fixture
    def collectors_dict(self, mocker: MockerFixture) -> dict[str, DataCollector]:
        return {
            "collector1": self.get_mock_data_collector(mocker, "data1"),
            "collector2": self.get_mock_data_collector(mocker, "data2"),
        }

    @pytest.fixture
    def dict_data_collectors(self, collectors_dict) -> DictDataCollectors:
        return DictDataCollectors(**collectors_dict)

    def test_init(self, dict_data_collectors, collectors_dict):
        assert dict_data_collectors["collector1"] is collectors_dict["collector1"]
        assert dict_data_collectors["collector2"] is collectors_dict["collector2"]

    def test_collect(self, dict_data_collectors: DictDataCollectors, collectors_dict):
        dict_data_collectors.collect({})

        for value in collectors_dict.values():
            value.collect.assert_called_once_with({})

    def test_get_data(self, dict_data_collectors: DictDataCollectors):
        assert dict_data_collectors.get_data() == {
            "collector1": "data1",
            "collector2": "data2",
        }
