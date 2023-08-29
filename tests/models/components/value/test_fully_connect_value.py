import pytest
import torch

from src.models.components.value.fully_connect_value import FullyConnectValue


@pytest.mark.parametrize(
    """
    batch,
    dim_input,
    """,
    [
        (4, 623),
        (32, 64),
        (17, 234),
    ],
)
def test_fully_connect_value(batch, dim_input):
    fcv = FullyConnectValue(dim_input)
    input = torch.randn(batch, dim_input)
    value = fcv(input)
    assert value.size() == (batch, 1)
