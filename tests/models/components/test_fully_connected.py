import pytest
import torch
import torch.nn as nn

from src.models.components.fully_connected import FullyConnected


@pytest.mark.parametrize(
    """
    batch,
    dim_in,
    dim_out,
    activation,
    """,
    [
        (4, 123, 345, None),
        (32, 234, 345, nn.ReLU()),
        (17, 234, 241, nn.ReLU()),
    ],
)
def test_fully_connected(batch, dim_in, dim_out, activation):
    fc = FullyConnected(dim_in, dim_out, activation=activation)
    x = torch.randn(batch, dim_in)
    x = fc(x)

    assert x.size() == (batch, dim_out)
