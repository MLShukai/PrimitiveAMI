import pytest
import torch

from src.models.components.stochastic_policy.normal_stochastic_policy import (
    NormalStochasticPolicy,
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
def test_normal_stochastic_policy(batch, dim_input, dim_out):
    nsp = NormalStochasticPolicy(dim_input, dim_out)
    input = torch.randn(batch, dim_input)
    dist = nsp(input)
    assert dist.rsample().size() == (batch, dim_out)
