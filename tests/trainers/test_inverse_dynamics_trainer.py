from functools import partial

import lightning.pytorch as pl
import pytest
import torch
from pytest_mock import MockerFixture
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.models.components.action_predictor.continuous_action_predictor import (
    ContinuousActionPredictor,
)
from src.models.components.inverse_dynamics.inverse_dynamics import InverseDynamics
from src.models.components.observation_encoder.cnn_observation_encoder import (
    CNNObservationEncoder,
)
from src.models.inverse_dynamics_lit_module import InverseDynamicsLitModule
from src.trainers.inverse_dynamics_trainer import InverseDynamicsTrainer as cls

num_batch = 32
obs_emb_dim = 256
action_dim = 128
num_channel = 3
width = 256
height = 256
obs_shape = (num_channel, width, height)


class TestInverseDynamicsTrainer:
    @pytest.fixture
    def obs_encoder(self):
        encoder = CNNObservationEncoder(obs_emb_dim, height, width, num_channel)
        return encoder

    @pytest.fixture
    def action_predictor(self):
        predictor = ContinuousActionPredictor(obs_emb_dim, action_dim)
        return predictor

    @pytest.fixture
    def inv_dynamics_net(self, obs_encoder, action_predictor):
        net = InverseDynamics(action_predictor, obs_encoder)
        return net

    @pytest.fixture
    def optimizer(self):
        optimizer = partial(Adam)
        return optimizer

    @pytest.fixture
    def inv_dynamics_lit_module(self, inv_dynamics_net, optimizer):
        module = InverseDynamicsLitModule(inv_dynamics_net, optimizer)
        return module

    @pytest.fixture
    def mock_dynamics_data_collector(self, mocker: MockerFixture):
        batch = (
            torch.randn((num_batch, action_dim)),
            torch.randn((num_batch, *obs_shape)),
            torch.randn((num_batch, action_dim)),
            torch.randn((num_batch, *obs_shape)),
        )
        mock = mocker.Mock(spec=DynamicsDataCollector)
        mock.get_data.return_value = TensorDataset(*batch)
        return mock

    @pytest.fixture
    def dataloader(self):
        return partial(DataLoader)

    def get_pl_trainer(self, accelerator):
        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        return trainer

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no available cuda")
    def test_cuda_train(self, inv_dynamics_lit_module, mock_dynamics_data_collector, dataloader):
        pl_trainer = self.get_pl_trainer("cuda")
        mod = cls(inv_dynamics_lit_module, mock_dynamics_data_collector, dataloader, pl_trainer)
        mod.train()
        mod.train()
        assert mod.pl_trainer.fit_loop.max_epochs == 3

    def test_cpu_train(self, inv_dynamics_lit_module, mock_dynamics_data_collector, dataloader):
        pl_trainer = self.get_pl_trainer("cpu")
        mod = cls(inv_dynamics_lit_module, mock_dynamics_data_collector, dataloader, pl_trainer)
        mod.train()
        mod.train()
        assert mod.pl_trainer.fit_loop.max_epochs == 3
