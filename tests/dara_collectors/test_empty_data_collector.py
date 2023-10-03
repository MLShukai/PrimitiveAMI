import pytest
from torch.utils.data import TensorDataset

from src.data_collectors.empty_data_collector import EmptyDataCollector


class TestEmptyDataCollector:
    @pytest.fixture
    def data_collector(self) -> EmptyDataCollector:
        return EmptyDataCollector()

    def test_collect(self, data_collector: EmptyDataCollector):
        data_collector.collect({})

    def test_get_data(self, data_collector: EmptyDataCollector):
        assert isinstance(data_collector.get_data(), TensorDataset)
        
    def test_state_dict(self, data_collector: EmptyDataCollector):
        assert data_collector.state_dict() == {}
    
    def test_load_state_dict(self, data_collector: EmptyDataCollector):
        data_collector.load_state_dict({})
