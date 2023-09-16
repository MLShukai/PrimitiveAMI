from functools import partial

import pytest
import torch
from pytest_mock import MockerFixture
from torch.optim import Adam, Optimizer

from src.models.components.forward_dynamics.forward_dynamics import ForwardDynamics
from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)
from src.models.forward_dynamics_lit_module import ForwardDynamicsLitModule


class TestForwardDynamicsLitModule:
    num_batch = 32
    obs_emb_dim = 256
    action_dim = 128
    batch = (
        torch.randn(num_batch, action_dim),
        torch.randn(num_batch, obs_emb_dim),
        torch.randn(num_batch, action_dim),
        torch.randn(num_batch, obs_emb_dim),
    )

    @pytest.fixture
    def mock_observation_encoder(self, mocker: MockerFixture) -> ObservationEncoder:
        mock = mocker.Mock(spec=ObservationEncoder, return_value=torch.randn(self.num_batch, self.obs_emb_dim))
        return mock

    @pytest.fixture
    def mock_forward_dynamics_net(self, mocker: MockerFixture) -> ForwardDynamics:
        mock = mocker.Mock(spec=ForwardDynamics, return_value=torch.randn(self.num_batch, self.obs_emb_dim))
        return mock

    @pytest.fixture
    def optimizer(self) -> partial[Optimizer]:
        return partial(Adam)

    @pytest.fixture
    def forward_dynamics_lit_module(
        self,
        mock_observation_encoder: ObservationEncoder,
        mock_forward_dynamics_net: ForwardDynamics,
        optimizer: Optimizer,
    ) -> ForwardDynamicsLitModule:
        return ForwardDynamicsLitModule(mock_observation_encoder, mock_forward_dynamics_net, optimizer)

    def test_forward_dynamics_lit_module(self, forward_dynamics_lit_module: ForwardDynamicsLitModule):
        loss = forward_dynamics_lit_module.training_step(self.batch, 0)
