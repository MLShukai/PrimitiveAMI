import pytest
from pytest_mock import MockerFixture

from src.trainers.sequential_trainers import SequentialTrainers as cls
from src.trainers.trainer import Trainer


class TestSequentialTrainers:
    num_trainers = 3

    @pytest.fixture
    def mock_trainers(self, mocker: MockerFixture):
        trainers = []
        for _ in range(self.num_trainers):
            mock = mocker.Mock(spec=Trainer)
            trainers.append(mock)
        return trainers

    def test__init__(self, mock_trainers):
        mod = cls(*mock_trainers)
        for input_trainers, attr_trainers in zip(mock_trainers, mod.trainers):
            assert input_trainers is attr_trainers

    def test_train(self, mock_trainers):
        mod = cls(*mock_trainers)
        mod.train()
        for trainer in mod.trainers:
            assert trainer.train.call_count == 1
