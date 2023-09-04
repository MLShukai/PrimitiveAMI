import pytest
import torch
from torch.distributions.distribution import Distribution

from src.models.components.policy.tanh_normal_stochastic_policy import (
    TanhNormalStochasticPolicy,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_input,
    dim_out,
    """,
    [
        (4, 623, 278),
        (32, 64, 125),
        (17, 234, 137),
    ],
)
def test_tanh_normal_stochastic_policy(batch, dim_input, dim_out):
    tnsp = TanhNormalStochasticPolicy(dim_input, dim_out)
    input = torch.randn(batch, dim_input)
    dist = tnsp(input)
    assert isinstance(dist, Distribution)
    assert dist.rsample().size() == (batch, dim_out)
    assert dist.entropy().size() == (batch, dim_out)
