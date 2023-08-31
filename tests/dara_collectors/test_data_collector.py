from src.data_collectors.data_collector import DataCollector


class DataCollectorImpl(DataCollector):
    def collect(self, step_record):
        pass


class TestDataCollector:
    def test_is_abstract(self):
        assert DataCollector.__abstractmethods__ == frozenset({"collect"})

    def test_collect(self):
        data_collector = DataCollectorImpl()
        data_collector.collect({})
