from functools import partial

import lightning.pytorch as pl
import pytest
import torch
from pytest_mock import MockerFixture
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.models.components.forward_dynamics.dense_net_forward_dynamics import (
    DenseNetForwardDynamics,
)
from src.models.components.forward_dynamics.forward_dynamics import ForwardDynamics
from src.models.components.forward_dynamics.forward_dynamics_lit_module import (
    ForwardDynamicsLitModule,
)
from src.models.components.observation_encoder.cnn_observation_encoder import (
    CNNObservationEncoder,
)
from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)
from src.trainers.forward_dynamics_trainer import ForwardDynamicsTrainer
from src.trainers.trainer import Trainer

num_batch = 32
obs_emb_dim = 256
action_dim = 128
channels = 3
height = 256
width = 256
obs_shape = (channels, height, width)
batch = (
    torch.randn(num_batch, action_dim),
    torch.randn(num_batch, obs_emb_dim),
    torch.randn(num_batch, action_dim),
    torch.randn(num_batch, obs_emb_dim),
)


@pytest.fixture
def mock_observation_encoder() -> ObservationEncoder:
    return CNNObservationEncoder(obs_emb_dim, height, width, channels)


@pytest.fixture
def mock_forward_dynamics_net() -> ForwardDynamics:
    return DenseNetForwardDynamics(action_dim, obs_emb_dim)


@pytest.fixture
def mock_optimizer(mock_forward_dynamics_net) -> partial[Optimizer]:
    return partial(Adam)


@pytest.fixture
def mock_forward_dynamics_lit_module(
    mock_observation_encoder: ObservationEncoder,
    mock_forward_dynamics_net: ForwardDynamics,
    mock_optimizer: Optimizer,
) -> ForwardDynamicsLitModule:
    return ForwardDynamicsLitModule(mock_observation_encoder, mock_forward_dynamics_net, mock_optimizer)


@pytest.fixture
def mock_dynamics_data_collector(mocker: MockerFixture) -> DynamicsDataCollector:
    mock = mocker.Mock(spec=DynamicsDataCollector)
    prev_actions = torch.randn(num_batch, action_dim)
    observations = torch.randn(num_batch, *obs_shape)
    actions = torch.randn(num_batch, action_dim)
    next_observations = torch.randn(num_batch, *obs_shape)
    mock.get_data.return_value = TensorDataset(prev_actions, observations, actions, next_observations)
    return mock


@pytest.fixture
def mock_dataloader() -> partial[DataLoader]:
    return partial(DataLoader)


@pytest.fixture
def mock_pl_trainer() -> pl.Trainer:
    return pl.Trainer(
        max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False
    )


@pytest.fixture
def forward_dynamics_trainer(
    mock_forward_dynamics_lit_module: ForwardDynamicsLitModule,
    mock_dynamics_data_collector: DynamicsDataCollector,
    mock_dataloader: partial[DataLoader],
    mock_pl_trainer: pl.Trainer,
):
    return ForwardDynamicsTrainer(
        mock_forward_dynamics_lit_module, mock_dynamics_data_collector, mock_dataloader, mock_pl_trainer
    )


def test_forward_dynamics_trainer(forward_dynamics_trainer):
    forward_dynamics_trainer.train()
