import pytest
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from src.models.components.policy.discrete_policy import DiscretePolicy


@pytest.mark.parametrize(
    """
    batch,
    dim_in,
    action_choices_per_category,
    """,
    [
        (8, 256, [3, 3, 3, 2, 2]),
        (1, 16, [1, 2, 3, 4, 5]),
    ],
)
def test_discrete_policy(batch, dim_in, action_choices_per_category):
    policy = DiscretePolicy(dim_in, action_choices_per_category)
    input = torch.randn(batch, dim_in)
    dist = policy(input)
    assert isinstance(dist, Distribution)
    assert dist.sample().shape == (batch, len(action_choices_per_category))
    assert dist.log_prob(dist.sample()).shape == (batch, len(action_choices_per_category))
    assert dist.entropy().shape == (batch, len(action_choices_per_category))
