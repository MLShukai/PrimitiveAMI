import pytest
import torch

from src.models.components.action_predictor.continuous_action_predictor import (
    ContinuousActionPredictor,
)
from src.models.components.inverse_dynamics.inverse_dynamics import InverseDynamics
from src.models.components.observation_encoder.cnn_observation_encoder import (
    CNNObservationEncoder,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_embed,
    dim_action,
    height,
    width,
    """,
    [
        (4, 8, 16, 32, 64),
        (32, 64, 125, 53, 47),
        (17, 234, 137, 349, 26),
    ],
)
def test_inverse_dynamics(batch, dim_embed, dim_action, height, width):
    action_precictor = ContinuousActionPredictor(dim_embed, dim_action)
    observation_encoder = CNNObservationEncoder(dim_embed, height, width)
    inverse_dynamics = InverseDynamics(action_precictor, observation_encoder)
    obs = torch.randn(batch, 3, height, width)
    next_obs = torch.randn(batch, 3, height, width)
    action_hat = inverse_dynamics(obs, next_obs)
    assert action_hat.size() == (batch, dim_action)
