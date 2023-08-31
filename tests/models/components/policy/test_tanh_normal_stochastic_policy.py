import pytest
import torch
from torch.distributions.distribution import Distribution

from src.models.components.policy.tahh_normal_stochastic_policy import (
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
    nsp = TanhNormalStochasticPolicy(dim_input, dim_out)
    input = torch.randn(batch, dim_input)
    dist = nsp(input)
    assert isinstance(dist, Distribution)
    assert dist.rsample().size() == (batch, dim_out)
