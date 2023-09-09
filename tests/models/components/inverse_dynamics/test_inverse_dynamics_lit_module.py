from copy import deepcopy

import pytest
import torch
from torch.optim import Adam

from src.models.components.inverse_dynamics.inverse_dynamics_lit_module import (
    InverseDynamicsLitModule as cls,
)

from ..dummy.inverse_dynamics.dummy_inverse_dynamics import DummyInverseDynamics

in_dim = 64
out_dim = 3


def make_batch(num_batch: int):
    batch = (
        torch.randn((num_batch, out_dim), requires_grad=True),
        torch.randn((num_batch, in_dim), requires_grad=True),
        torch.randn((num_batch, out_dim), requires_grad=True),
        torch.randn((num_batch, in_dim), requires_grad=True),
    )
    return batch


params = [[make_batch(1), 0], [make_batch(8), 1]]


class TestInverseDynamicsLitModule:
    @pytest.fixture
    def dummy_net(self):
        return DummyInverseDynamics(in_dim, out_dim)

    @pytest.fixture
    def optimizer(self, dummy_net):

        return Adam(dummy_net.parameters())

    def test__init__(self, dummy_net, optimizer):
        mod = cls(dummy_net, optimizer)
        assert mod.hparams.inverse_dynamics_net is dummy_net
        assert mod.hparams.optimizer is optimizer

    def test_configure_optimizers(self, dummy_net, optimizer):
        mod = cls(dummy_net, optimizer)
        assert mod.configure_optimizers() is optimizer

    @pytest.mark.parametrize("batch, batch_idx", params)
    def test_training_step(self, dummy_net, optimizer, batch, batch_idx):
        mod = cls(dummy_net, optimizer)
        init_parameters = [param.clone() for param in mod.hparams.inverse_dynamics_net.parameters()]
        loss = mod.training_step(batch, batch_idx)
        loss.backward()
        mod.hparams.optimizer.step()
        after_parameters = [param.clone() for param in mod.hparams.inverse_dynamics_net.parameters()]

        # モデルパラメータの更新前後を比較
        for init_param, after_param in zip(init_parameters, after_parameters):
            assert not torch.all(
                torch.eq(
                    init_param,
                    after_param,
                )
            ), "Model parameters were not updated."
