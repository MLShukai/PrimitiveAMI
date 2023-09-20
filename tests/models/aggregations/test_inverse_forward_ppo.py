from functools import partial

import pytest
from pytest_mock import MockerFixture

from src.models.aggregations.inverse_forward_ppo import (
    ForwardDynamicsLitModule,
    InverseDynamicsLitModule,
    InverseForwardPPO,
    PPOLitModule,
)


class TestInverseForwardPPO:
    @pytest.fixture
    def mock_inv_dyna_lit(self, mocker: MockerFixture) -> InverseDynamicsLitModule:
        mock = mocker.Mock(spec=InverseDynamicsLitModule)
        mock.net = mocker.Mock()
        mock.net.observation_encoder = mocker.Mock()
        return mock

    @pytest.fixture
    def mock_for_dyna_lit(self, mocker: MockerFixture) -> ForwardDynamicsLitModule:
        mock = mocker.Mock(spec=ForwardDynamicsLitModule)
        mock.forward_dynamics_net = mocker.Mock()
        return partial(mock)

    @pytest.fixture
    def mock_ppo_lit(self, mocker: MockerFixture) -> PPOLitModule:
        mock = mocker.Mock(spec=PPOLitModule)
        mock.net = mocker.Mock()
        return mock

    @pytest.fixture
    def inverse_forward_ppo(self, mock_inv_dyna_lit, mock_for_dyna_lit, mock_ppo_lit) -> InverseForwardPPO:
        return InverseForwardPPO(mock_inv_dyna_lit, mock_for_dyna_lit, mock_ppo_lit)

    def test_init(self, inverse_forward_ppo: InverseForwardPPO, mock_inv_dyna_lit, mock_for_dyna_lit, mock_ppo_lit):
        assert inverse_forward_ppo.inverse_dynamics is mock_inv_dyna_lit
        assert inverse_forward_ppo.forward_dynamics is mock_for_dyna_lit(mock_inv_dyna_lit.net.observation_encoder)
        assert inverse_forward_ppo.ppo is mock_ppo_lit

    def test_build_agent_models(self, inverse_forward_ppo: InverseForwardPPO):
        models = inverse_forward_ppo.build_agent_models()
        assert isinstance(models, dict)
        for keyword in models.keys():
            assert isinstance(keyword, str)
