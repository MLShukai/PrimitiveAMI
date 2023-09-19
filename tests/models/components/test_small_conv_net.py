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
    do_batchnorm,
    do_layernorm,
    nl,
    last_nl
    """,
    [
        (4, 123, 345, 3, 623, False, False, torch.nn.ReLU(), None),
        (32, 234, 345, 2, 64, True, False, None, torch.nn.ReLU()),
        (17, 234, 241, 1, 345, False, True, None, torch.nn.LeakyReLU()),
    ],
)
def test_small_conv_net(batch, height, width, channels, dim_out, do_batchnorm, do_layernorm, nl, last_nl):
    if nl is not None:
        conv_net = SmallConvNet(height, width, channels, dim_out, do_batchnorm, do_layernorm, nl, last_nl)
    else:
        conv_net = SmallConvNet(height, width, channels, dim_out, do_batchnorm, do_layernorm, last_nl=last_nl)
    x = torch.randn(batch, channels, height, width)
    y = conv_net(x)
    assert y.size() == (batch, dim_out)
