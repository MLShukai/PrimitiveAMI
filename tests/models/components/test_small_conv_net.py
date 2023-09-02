import pytest
import torch

from src.models.components.small_conv_net import SmallConvNet


@pytest.mark.parametrize(
    """
    batch,
    height,
    width,
    dim_out,
    """,
    [
        (4, 123, 345, 623),
        (32, 234, 345, 64),
        (17, 234, 241, 345),
    ],
)
def test_small_conv_net(batch, height, width, dim_out):
    conv_net = SmallConvNet(height, width, dim_out)
    x = torch.randn(batch, 3, height, width)
    y = conv_net(x)
    assert y.size() == (batch, dim_out)
