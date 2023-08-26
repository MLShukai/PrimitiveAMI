import pytest
import torch

from src.models.components.observation_encoder.cnn_observation_encoder import (
    CNNObservationEncoder,
)


@pytest.mark.parametrize(
    """
    batch,
    dim_embed,
    height,
    width,
    """,
    [
        (4, 8, 16, 32),
        (32, 64, 125, 53),
        (17, 234, 137, 349),
    ],
)
def test_cnn_obeservation_encoder(batch, dim_embed, height, width):
    cnn_observation_encoder = CNNObservationEncoder(dim_embed, height, width)
    x = torch.randn(batch, 3, height, width)
    x = cnn_observation_encoder(x)
    assert x.size() == (batch, dim_embed)
