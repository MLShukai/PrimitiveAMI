import pytest
import torch
from torch.distributions.distribution import Distribution

from src.models.components.reward.curiosity_reward import CuriosityReward


@pytest.mark.parametrize(
    """
    batch,
    dim,
    """,
    [
        (4, 623),
        (32, 64),
        (17, 234),
    ],
)
def test_tanh_normal_stochastic_policy(batch, dim):
    obs_hat = torch.randn(batch, dim)
    obs = torch.randn(batch, dim)
    curiosity_reward = CuriosityReward()
    reward = curiosity_reward(obs_hat, obs)
    assert reward.size() == (batch,)
