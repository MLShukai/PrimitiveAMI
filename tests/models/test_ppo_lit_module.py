from functools import partial

import pytest
import torch
from pytest_mock import MockerFixture
from torch.distributions import Normal
from torchmetrics import MeanMetric

from src.models.ppo_lit_module import PolicyValueCommonNet, PPOLitModule


class TestPPOLitModule:
    batch_size = 8
    action_shape = (3,)
    obs_shape = (3, 256, 256)

    @pytest.fixture
    def mock_net(self, mocker: MockerFixture) -> PolicyValueCommonNet:
        action_dist = Normal(
            torch.zeros(self.batch_size, *self.action_shape), torch.ones(self.batch_size, *self.action_shape)
        )
        value = torch.randn(self.batch_size, 1)
        mock = mocker.Mock(spec=PolicyValueCommonNet, return_value=(action_dist, value))

        return mock

    @pytest.fixture
    def ppo_lit_module(self, mock_net) -> PPOLitModule:
        optim = partial(torch.optim.Adam, lr=0.001)
        return PPOLitModule(mock_net, optim)

    @pytest.fixture
    def batch(self) -> tuple[torch.Tensor, ...]:
        obses = torch.randn(self.batch_size, *self.obs_shape)
        actions = torch.tanh(torch.randn(self.batch_size, *self.action_shape))
        logprobs = -torch.randn_like(actions).abs()
        advantages = torch.randn(self.batch_size)
        returns = torch.randn_like(advantages)
        values = torch.randn_like(advantages)

        return obses, actions, logprobs, advantages, returns, values

    def test_init(self, ppo_lit_module: PPOLitModule):

        assert isinstance(ppo_lit_module.net, PolicyValueCommonNet)

    def test_model_step(self, ppo_lit_module: PPOLitModule, batch: tuple[torch.Tensor, ...]):

        output = ppo_lit_module.model_step(batch)

        assert "loss" in output
        assert "value_loss" in output
        assert "policy_loss" in output
        assert "clipfrac" in output

        loss = output["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == ()
