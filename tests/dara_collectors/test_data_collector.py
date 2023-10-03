import pickle

import pytest

from src.data_collectors.data_collector import DataCollector


class DataCollectorImpl(DataCollector):
    def __init__(self, kwds1=0):
        self.kwds1 = kwds1

    def collect(self, step_record):
        pass

    def get_data(self):
        return 0

    def state_dict(self) -> dict[str, int]:
        return {"data": 0}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        assert state_dict["data"] == 0


class TestDataCollector:
    @pytest.fixture
    def example_data_collector(self):
        return DataCollectorImpl()

    def test_is_abstract(self):
        assert DataCollector.__abstractmethods__ == frozenset({"collect", "get_data", "state_dict", "load_state_dict"})

    def test_collect(self, example_data_collector):
        example_data_collector.collect({})
        assert example_data_collector.get_data() == 0

    def test_save_to_file(self, tmp_path, example_data_collector):
        dist_path = tmp_path / "data_collector.pkl"

        # Save to file
        example_data_collector.save_to_file(dist_path)

        # Ensure file is created
        assert dist_path.exists()
        assert dist_path.is_file()

        # Ensure saved data is as expected
        with open(dist_path, mode="rb") as f:
            assert pickle.load(f) == {"data": 0}

    def test_load_from_file(self, tmp_path, example_data_collector):
        dist_path = tmp_path / "data_collector.pkl"

        # Save to file
        example_data_collector.save_to_file(dist_path)

        # Load from file
        loaded_data_collector = DataCollectorImpl.load_from_file(dist_path, kwds1=1)

        # Ensure loaded data is as expected
        assert loaded_data_collector.get_data() == 0

        assert loaded_data_collector.kwds1 == 1
