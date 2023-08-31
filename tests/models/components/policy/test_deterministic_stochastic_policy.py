import pytest
import torch
from torch.distributions.distribution import Distribution

from src.models.components.policy.deterministic_stochastic_policy import (
    DeterministicStochasticPolicy,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_input,
    """,
    [
        (4, 623),
        (32, 64),
        (17, 234),
    ],
)
def test_deterministic_stochastic_policy(batch, dim_input):
    dsp = DeterministicStochasticPolicy()
    input = torch.randn(batch, dim_input)
    dist = dsp(input)
    assert isinstance(dist, Distribution)
    assert dist.rsample().size() == (batch, dim_input)
    assert torch.isclose(dist.rsample(), input).all()
