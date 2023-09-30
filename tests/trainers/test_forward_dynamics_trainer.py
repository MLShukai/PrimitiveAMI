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
from src.models.components.observation_encoder.cnn_observation_encoder import (
    CNNObservationEncoder,
)
from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)
from src.models.forward_dynamics_lit_module import ForwardDynamicsLitModule
from src.trainers.forward_dynamics_trainer import ForwardDynamicsTrainer


class TestForwardDynamicsTrainer:
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
    def observation_encoder(self) -> ObservationEncoder:
        return CNNObservationEncoder(self.obs_emb_dim, self.height, self.width, self.channels)

    @pytest.fixture
    def forward_dynamics_net(self) -> ForwardDynamics:
        return DenseNetForwardDynamics(self.action_dim, self.obs_emb_dim)

    @pytest.fixture
    def optimizer(self) -> partial[Optimizer]:
        return partial(Adam)

    @pytest.fixture
    def forward_dynamics_lit_module(
        self,
        observation_encoder: ObservationEncoder,
        forward_dynamics_net: ForwardDynamics,
        optimizer: Optimizer,
    ) -> ForwardDynamicsLitModule:
        return ForwardDynamicsLitModule(observation_encoder, forward_dynamics_net, optimizer)

    @pytest.fixture
    def mock_dynamics_data_collector(self, mocker: MockerFixture) -> DynamicsDataCollector:
        mock = mocker.Mock(spec=DynamicsDataCollector)
        prev_actions = torch.randn(self.num_batch, self.action_dim)
        observations = torch.randn(self.num_batch, *(self.obs_shape))
        actions = torch.randn(self.num_batch, self.action_dim)
        next_observations = torch.randn(self.num_batch, *(self.obs_shape))
        mock.get_data.return_value = TensorDataset(prev_actions, observations, actions, next_observations)
        return mock

    @pytest.fixture
    def dataloader(self) -> partial[DataLoader]:
        return partial(DataLoader)

    @pytest.fixture
    def pl_trainer(self) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    @pytest.fixture
    def forward_dynamics_trainer(
        self,
        forward_dynamics_lit_module: ForwardDynamicsLitModule,
        mock_dynamics_data_collector: DynamicsDataCollector,
        dataloader: partial[DataLoader],
        pl_trainer: pl.Trainer,
    ):
        return ForwardDynamicsTrainer(forward_dynamics_lit_module, mock_dynamics_data_collector, dataloader, pl_trainer)

    def test_forward_dynamics_trainer(self, forward_dynamics_trainer):
        forward_dynamics_trainer.train()
        forward_dynamics_trainer.train()
        assert forward_dynamics_trainer.pl_trainer.fit_loop.max_epochs == 3
