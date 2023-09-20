import pytest
import torch

from src.models.components.action_predictor.continuous_action_predictor import (
    ContinuousActionPredictor,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_embed,
    dim_action,
    dim_hidden
    """,
    [
        (4, 8, 16, None),
        (32, 64, 128, 128),
    ],
)
def test_continuous_action_predictor(batch, dim_embed, dim_action, dim_hidden):
    continuous_action_predictor = ContinuousActionPredictor(dim_embed, dim_action, dim_hidden)
    embed = torch.randn(batch, dim_embed)
    embed_next = torch.randn(batch, dim_embed)
    prev_action_hat, action_hat = continuous_action_predictor(embed, embed_next)
    assert prev_action_hat.size() == (batch, dim_action)
    assert action_hat.size() == (batch, dim_action)
