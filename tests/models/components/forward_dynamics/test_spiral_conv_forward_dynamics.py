import pytest
import torch

from src.models.components.forward_dynamics.spiral_conv_forward_dynamics import (
    SpiralConvForwardDynamics,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_action,
    dim_embed,
    depth,
    dim,
    dim_ff_scale,
    dropout
    """,
    [
        (4, 623, 278, 2, 16, 2, 0.1),
    ],
)
def test_spiral_conv_forward_dynamics(batch, dim_action, dim_embed, depth, dim, dim_ff_scale, dropout):
    scfd = SpiralConvForwardDynamics(dim_action, dim_embed, depth, dim, dim_ff_scale, dropout)
    prev_action = torch.randn(batch, dim_action)
    embed = torch.randn(batch, dim_embed)
    action = torch.randn(batch, dim_action)
    next_embed = scfd(prev_action, embed, action)
    assert next_embed.size() == (batch, dim_embed)
