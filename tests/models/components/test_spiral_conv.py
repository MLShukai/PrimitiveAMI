import pytest
import torch

from src.models.components.spiral_conv import Architecture


class TestSpiralConv:
    @pytest.mark.parametrize(
        """
        depth,
        dim,
        dim_ff_scale,
        dropout,
        batch,
        length,
        """,
        [
            (4, 16, 2, 0.1, 64, 128),
        ],
    )
    def test_spiral_conv(self, depth, dim, dim_ff_scale, dropout, batch, length):
        model = Architecture(depth, dim, dim_ff_scale, dropout)
        x = torch.randn(batch, length, dim)
        x, hidden_list = model(x)
        assert x.size() == (batch, length, dim)
        assert len(hidden_list) == depth
        for hidden in hidden_list:
            assert hidden.size() == (batch, length, dim)
