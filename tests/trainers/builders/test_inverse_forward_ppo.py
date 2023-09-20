import pytest
from pytest_mock import MockerFixture

from src.trainers.builders.inverse_forward_ppo import (
    DictDataCollectors,
    ForwardDynamicsTrainer,
    InverseDynamicsTrainer,
    InverseForwardPPO,
    PPOTrainer,
    SequentialTrainers,
    neuralnets,
)


class TestInverseForwardPPO:
    @pytest.fixture
    def mock_inverse_dynamics(self, mocker: MockerFixture):
        return mocker.Mock(spec=InverseDynamicsTrainer)

    @pytest.fixture
    def mock_forward_dynamics(self, mocker: MockerFixture):
        return mocker.Mock(spec=ForwardDynamicsTrainer)

    @pytest.fixture
    def mock_ppo(self, mocker: MockerFixture):
        return mocker.Mock(spec=PPOTrainer)

    @pytest.fixture
    def inverse_forward_ppo(self, mock_inverse_dynamics, mock_forward_dynamics, mock_ppo):
        return InverseForwardPPO(mock_inverse_dynamics, mock_forward_dynamics, mock_ppo)

    @pytest.fixture
    def mock_nets(self, mocker: MockerFixture):
        mock = mocker.Mock(spec=neuralnets.InverseForwardPPO)
        mock.inverse_dynamics = mocker.Mock()
        mock.forward_dynamics = mocker.Mock()
        mock.ppo = mocker.Mock()
        return mock

    @pytest.fixture
    def mock_data_collectors(self, mocker: MockerFixture):
        return DictDataCollectors(dynamics=mocker.Mock(), trajectory=mocker.Mock())

    def test_init(self, inverse_forward_ppo, mock_inverse_dynamics, mock_forward_dynamics, mock_ppo):
        assert inverse_forward_ppo.inverse_dynamics is mock_inverse_dynamics
        assert inverse_forward_ppo.forward_dynamics is mock_forward_dynamics
        assert inverse_forward_ppo.ppo is mock_ppo

    def test_build(self, inverse_forward_ppo, mock_nets, mock_data_collectors):
        trainers = inverse_forward_ppo.build(mock_nets, mock_data_collectors)
        assert isinstance(trainers, SequentialTrainers)
        assert trainers.trainers == (
            inverse_forward_ppo.inverse_dynamics.return_value,
            inverse_forward_ppo.forward_dynamics.return_value,
            inverse_forward_ppo.ppo.return_value,
        )
