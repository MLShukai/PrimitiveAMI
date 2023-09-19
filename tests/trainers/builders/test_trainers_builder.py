from src.trainers.builders.trainers_builder import TrainersBuilder


class TestTrainersBuilder:
    def test_is_abstract(self):
        assert TrainersBuilder.__abstractmethods__ == frozenset({"build"})
