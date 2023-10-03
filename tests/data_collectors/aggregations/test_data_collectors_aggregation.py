from src.data_collectors.aggregations.data_collectors_aggregation import (
    DataCollectorsAggregation,
)


class TestDataCollectorsAggregation:
    def test_is_abstract(self):
        assert DataCollectorsAggregation.__abstractmethods__ == frozenset({"collect", "save_state_dict_to_files"})
