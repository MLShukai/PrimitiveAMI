import pytest
import torch

from src.models.components.forward_dynamics.spiral_conv_forward_dynamics import (
    DiscreteActionSCFD,
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


@pytest.mark.parametrize(
    """
    batch,
    action_choices_per_category,
    action_embedding_dim,
    dim_embed,
    """,
    [
        (8, [3, 5, 2], 16, 512),
        (4, [9, 4], 32, 256),
    ],
)
def test_discrete_action_scfd(batch, action_choices_per_category, action_embedding_dim, dim_embed):
    dascfd = DiscreteActionSCFD(
        action_choices_per_category,
        action_embedding_dim,
        dim_embed,
        4,
        512,
        2,
        0.1,
    )

    prev_action = torch.stack([torch.randint(0, i, (batch,)) for i in action_choices_per_category], -1)
    embed = torch.randn(batch, dim_embed)
    action = torch.stack([torch.randint(0, i, (batch,)) for i in action_choices_per_category], -1)

    next_embed = dascfd(prev_action, embed, action)
    assert next_embed.size() == (batch, dim_embed)
