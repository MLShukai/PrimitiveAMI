import pytest

from src.data_collectors.empty_data_collector import EmptyDataCollector


class TestEmptyDataCollector:
    @pytest.fixture
    def data_collector(self) -> EmptyDataCollector:
        return EmptyDataCollector()

    def test_collect(self, data_collector: EmptyDataCollector):
        data_collector.collect({})

    def test_get_data(self, data_collector: EmptyDataCollector):
        assert data_collector.get_data() is None
