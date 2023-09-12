from functools import partial

import pytest
import torch
from pytest_mock import MockerFixture
from torch.optim import Adam, Optimizer

from src.models.components.forward_dynamics.dense_net_forward_dynamics import (
    DenseNetForwardDynamics,
)
from src.models.components.forward_dynamics.forward_dynamics import ForwardDynamics
from src.models.components.forward_dynamics.forward_dynamics_lit_module import (
    ForwardDynamicsLitModule,
)
from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)

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
def mock_observation_encoder(mocker: MockerFixture) -> ObservationEncoder:
    mock = mocker.Mock(spec=ObservationEncoder, return_value=torch.randn(num_batch, obs_emb_dim))
    return mock


@pytest.fixture
def mock_forward_dynamics_net() -> ForwardDynamics:
    return DenseNetForwardDynamics(action_dim, obs_emb_dim)


@pytest.fixture
def mock_optimizer(mock_forward_dynamics_net) -> partial[Optimizer]:
    return partial(Adam)


@pytest.fixture
def forward_dynamics_lit_module(
    mock_observation_encoder: ObservationEncoder,
    mock_forward_dynamics_net: ForwardDynamics,
    mock_optimizer: Optimizer,
) -> ForwardDynamicsLitModule:
    return ForwardDynamicsLitModule(mock_observation_encoder, mock_forward_dynamics_net, mock_optimizer)


def test_forward_dynamics_lit_module(forward_dynamics_lit_module: ForwardDynamicsLitModule):
    loss = forward_dynamics_lit_module.training_step(batch, 0)
