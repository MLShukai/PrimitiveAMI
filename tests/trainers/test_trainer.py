from src.trainers.trainer import Trainer


class TestTrainer:
    def test_is_abstract(self):
        assert Trainer.__abstractmethods__ == frozenset({"train"})
