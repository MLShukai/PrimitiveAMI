from src.data_collectors.data_collector import DataCollector


class DataCollectorImpl(DataCollector):
    def collect(self, step_record):
        pass

    def get_data(self):
        return 0


class TestDataCollector:
    def test_is_abstract(self):
        assert DataCollector.__abstractmethods__ == frozenset({"collect", "get_data"})

    def test_collect(self):
        data_collector = DataCollectorImpl()
        data_collector.collect({})
        assert data_collector.get_data() == 0
