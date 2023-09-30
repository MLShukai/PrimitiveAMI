from functools import partial

import lightning.pytorch as pl
import pytest
import torch
import torch.distributions as distributions
import torch.nn as nn
from pytest_mock import MockerFixture
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.data_collectors.trajectory_data_collector import TrajectoryDataCollector
from src.models.components.policy.normal_stochastic_policy import NormalStochasticPolicy
from src.models.components.policy.stochastic_policy import StochasticPolicy
from src.models.components.policy_value_common_net import PolicyValueCommonNet
from src.models.components.value.fully_connect_value import FullyConnectValue
from src.models.components.value.value import Value
from src.models.ppo_lit_module import PPOLitModule
from src.trainers.ppo_trainer import PPOTrainer


class TestPPOTrainer:
    num_batch = 64
    hidden_dim = 128
    action_dim = 128
    observation_dim = 64

    @pytest.fixture
    def base_model(self) -> nn.Module:
        return nn.Linear(self.observation_dim, self.hidden_dim)

    @pytest.fixture
    def policy(self) -> StochasticPolicy:
        return NormalStochasticPolicy(self.hidden_dim, self.action_dim)

    @pytest.fixture
    def value(self) -> Value:
        return FullyConnectValue(self.hidden_dim)

    @pytest.fixture
    def policy_value_common_net(self, base_model, policy, value) -> PolicyValueCommonNet:
        return PolicyValueCommonNet(base_model, policy, value)

    @pytest.fixture
    def optimizer(self) -> partial[Optimizer]:
        return partial(Adam)

    @pytest.fixture
    def ppo_lit_module(self, policy_value_common_net: PolicyValueCommonNet, optimizer: Optimizer) -> PPOLitModule:
        return PPOLitModule(policy_value_common_net, optimizer)

    @pytest.fixture
    def mock_data_collector(self, mocker: MockerFixture) -> TrajectoryDataCollector:
        mock = mocker.Mock(spec=TrajectoryDataCollector)
        observations = torch.randn(self.num_batch, self.observation_dim)
        actions = torch.randn(self.num_batch, self.action_dim)
        logprobs = torch.randn(self.num_batch)
        advantages = torch.randn(self.num_batch)
        values = torch.randn(self.num_batch)
        returns = advantages + values
        mock.get_data.return_value = TensorDataset(
            observations,
            actions,
            logprobs,
            advantages,
            returns,
            values,
        )
        return mock

    @pytest.fixture
    def dataloader(self) -> partial[DataLoader]:
        return partial(DataLoader)

    @pytest.fixture
    def logger(self, tmp_path):
        return pl.loggers.TensorBoardLogger(tmp_path / "tensorboard", name=None, version="")

    @pytest.fixture
    def pl_trainer(self, logger) -> pl.Trainer:
        return pl.Trainer(
            max_steps=10,
            logger=logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    @pytest.fixture
    def ppo_trainer(
        self,
        ppo_lit_module: PPOLitModule,
        mock_data_collector: TrajectoryDataCollector,
        dataloader: partial[DataLoader],
        pl_trainer: pl.Trainer,
    ):
        return PPOTrainer(ppo_lit_module, mock_data_collector, dataloader, pl_trainer)

    def test_ppo_trainer(self, ppo_trainer: PPOTrainer):
        ppo_trainer.train()
        ppo_trainer.train()
        assert ppo_trainer.pl_trainer.fit_loop.max_epochs == 3
