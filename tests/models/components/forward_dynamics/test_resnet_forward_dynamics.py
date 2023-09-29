import pytest
import torch

from src.models.components.forward_dynamics.resnet_forward_dynamics import (
    ResNetForwardDynamics,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_action,
    dim_embed,
    dim_hidden,
    depth,
    """,
    [
        (4, 623, 278, 234, 2),
        (32, 64, 125, 345, 3),
        (17, 234, 137, 514, 4),
    ],
)
def test_resnet_forward_dynamics(batch, dim_action, dim_embed, dim_hidden, depth):
    rfd = ResNetForwardDynamics(dim_action, dim_embed, dim_hidden, depth)
    prev_action = torch.randn(batch, dim_action)
    embed = torch.randn(batch, dim_embed)
    action = torch.randn(batch, dim_action)
    next_embed = rfd(prev_action, embed, action)
    assert next_embed.size() == (batch, dim_embed)
