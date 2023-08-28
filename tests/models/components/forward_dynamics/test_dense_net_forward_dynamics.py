import pytest
import torch

from src.models.components.forward_dynamics.dense_net_forward_dynamics import (
    DenseNetForwardDynamics,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_action,
    dim_embed,
    """,
    [
        (4, 623, 278),
        (32, 64, 125),
        (17, 234, 137),
    ],
)
def test_dense_net_forward_dynamics(batch, dim_action, dim_embed):
    fd = DenseNetForwardDynamics(dim_action, dim_embed)
    prev_action = torch.randn(batch, dim_action)
    embed = torch.randn(batch, dim_embed)
    action = torch.randn(batch, dim_action)
    next_embed = fd(prev_action, embed, action)
    assert next_embed.size() == (batch, dim_embed)
