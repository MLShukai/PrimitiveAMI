import pytest
import torch

from src.models.components.fully_connected import FullyConnected


@pytest.mark.parametrize(
    """
    batch,
    dim_in,
    dim_out,
    """,
    [
        (4, 123, 345),
        (32, 234, 345),
        (17, 234, 241),
    ],
)
def test_fully_connected(batch, dim_in, dim_out):
    fc = FullyConnected(dim_in, dim_out)
    x = torch.randn(batch, dim_in)
    x = fc(x)

    assert x.size() == (batch, dim_out)
