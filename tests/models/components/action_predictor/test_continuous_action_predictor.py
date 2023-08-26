import pytest
import torch

from src.models.components.action_predictor.continuous_action_predictor import ContinuousActionPredictor


@pytest.mark.parametrize(
    """
    batch,
    dim_embed,
    dim_action,
    """,
    [
        (4,8,16),
        (32,64,128),
    ]
)
def test_continuous_action_predictor(batch, dim_embed, dim_action):
    continuous_action_predictor = ContinuousActionPredictor(dim_embed, dim_action)
    embed = torch.randn(batch, dim_embed)
    embed_next = torch.randn(batch, dim_embed)
    action = torch.randn(batch, dim_action)
    loss, action_hat = continuous_action_predictor(embed, embed_next, action)
    assert loss.size() == ()
    assert action_hat.size() == (batch, dim_action)