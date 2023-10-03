import os
from pathlib import Path
from typing import Any

import pytest

from src.data_collectors.aggregations.dict_data_collectors import DictDataCollectors
from src.data_collectors.empty_data_collector import EmptyDataCollector


class TestDictDataCollectors:
    @pytest.fixture
    def collectors_dict(self) -> dict[str, EmptyDataCollector]:
        return {
            "collector1": EmptyDataCollector(),
            "collector2": EmptyDataCollector(),
        }

    @pytest.fixture
    def dict_data_collectors(self, collectors_dict) -> DictDataCollectors:
        return DictDataCollectors(**collectors_dict)

    def test_init(self, dict_data_collectors, collectors_dict):
        assert dict_data_collectors["collector1"] is collectors_dict["collector1"]
        assert dict_data_collectors["collector2"] is collectors_dict["collector2"]

    def test_collect(self, dict_data_collectors: DictDataCollectors):
        dict_data_collectors.collect({})

    def test_save_state_dict_to_files(self, tmp_path: Path, dict_data_collectors: DictDataCollectors):
        dict_data_collectors.save_state_dict_to_files(tmp_path)

        assert set(os.listdir(tmp_path)) == {"collector1.pkl", "collector2.pkl"}
