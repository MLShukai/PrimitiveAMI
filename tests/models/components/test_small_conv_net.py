import pytest
import torch

from src.models.components.small_conv_net import SmallConvNet


@pytest.mark.parametrize(
    """
    batch,
    height,
    width,
    channels,
    dim_out,
    """,
    [
        (4, 123, 345, 3, 623),
        (32, 234, 345, 2, 64),
        (17, 234, 241, 1, 345),
    ],
)
def test_small_conv_net(batch, height, width, channels, dim_out):
    conv_net = SmallConvNet(height, width, channels, dim_out)
    x = torch.randn(batch, channels, height, width)
    y = conv_net(x)
    assert y.size() == (batch, dim_out)
