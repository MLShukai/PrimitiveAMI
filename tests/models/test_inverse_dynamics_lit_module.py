from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.optim import Adam

from src.models.components.inverse_dynamics.inverse_dynamics import InverseDynamics
from src.models.inverse_dynamics_lit_module import InverseDynamicsLitModule as cls


class TestInverseDynamicsLitModule:
    batch_size = 8
    in_shape = (batch_size, 3, 256, 256)
    out_shape = (batch_size, 3)

    @pytest.fixture
    def batch(self):
        batch = (
            torch.randn(self.out_shape, requires_grad=True),
            torch.randn(self.in_shape, requires_grad=True),
            torch.randn(self.out_shape, requires_grad=True),
            torch.randn(self.in_shape, requires_grad=True),
        )
        return batch

    @pytest.fixture
    def dummy_net(self, mocker):
        dummy = mocker.Mock(
            spec=InverseDynamics, return_value=(torch.empty(self.out_shape), torch.empty(self.out_shape))
        )
        return dummy

    @pytest.fixture
    def optimizer(self):
        return partial(Adam)

    def test__init__(self, dummy_net, optimizer):
        mod = cls(dummy_net, optimizer)
        assert mod.net is dummy_net
        assert mod.hparams.optimizer is optimizer

    def test_training_step(self, dummy_net, optimizer, batch):
        mod = cls(dummy_net, optimizer)
        assert type(mod(batch, 0) == torch.Tensor)

    def test_forward(self, dummy_net, optimizer, batch):
        mod = cls(dummy_net, optimizer)
        _, obs, _, next_obs = batch
        output = mod.forward(obs, next_obs)
        assert len(output) == 2
        for out in output:
            assert out.shape == torch.Size(self.out_shape)
