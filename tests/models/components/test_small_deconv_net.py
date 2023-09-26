import pytest
import torch

from src.models.components.small_deconv_net import SmallDeconvNet as cls


class TestSmallDeconvNet:
    
    def test__init__(self):
        params = {
            "height": 256,
            "width": 256,
            "channels": 3,
            "dim_in": 256,
            "positional_bias": True,
            "nl": torch.nn.ReLU()
        }
        mod = cls(**params)
        assert mod.channels == params["channels"]
        assert mod.fc_init.in_features == params["dim_in"]
        assert mod.bias.shape == (params["channels"], params["height"], params["width"])
        assert mod.nl == params["nl"]

    @pytest.mark.parametrize(
            """batch,
            height,
            width,
            channels,
            dim_in,
            positional_bias,
            nl""",
            [
                (8, 256, 256, 3, 256, False, torch.nn.LeakyReLU()),
                (1, 256, 256, 3, 128, True, torch.nn.LeakyReLU(negative_slope=0.2))
            ]
    )
    def test_forward(self, batch, height, width, channels, dim_in, positional_bias, nl):
        mod = cls(height, width, channels, dim_in, positional_bias, nl)
        x = torch.randn(batch, dim_in)
        x = mod.forward(x)
        assert x.size(0) == batch, "batch size missmatch"
        assert x.size(1) == channels, "channel size missmatch"
        assert x.size(2) == height, "height missmatch"
        assert x.size(3) == width, "width missmatch"

