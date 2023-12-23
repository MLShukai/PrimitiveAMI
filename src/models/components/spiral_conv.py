import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .stacked_hidden_state import StackedHiddenState


class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * dim_ff_scale, bias=True)
        self.linear_2 = nn.Linear(dim * dim_ff_scale, dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class SpiralConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))  # log(-log(gamma))
        self.phazor = nn.Parameter(torch.exp(2.0j * np.pi * torch.arange(dim) / dim) * torch.abs(torch.randn(dim)))

    # ((batch, len, dim),(batch, dim)) -> ((batch, len, dim), (batch, len, dim))
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        len = x.shape[1]
        phazor = self.phazor / self.phazor.abs() * torch.exp(-self.phazor.abs())
        phazor_progression = torch.pow(
            phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)
        )  # (len, dim)
        filter = phazor_progression * self.phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len * 2, dim=0)  # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len * 2, dim=1)  # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1, 0, len)  # (batch, len, dim)
        conv_with_past = conv_filter_x + hidden.detach().unsqueeze(1) * phazor_progression.unsqueeze(
            0
        ) * phazor.unsqueeze(0).unsqueeze(0)

        return conv_with_past.real, conv_with_past


class ArchitectureBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.spiral_conv = SpiralConv(dim)
        self.ffn = FFN(dim, dim_ff_scale)
        self.layer_norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        y = x
        x = self.layer_norm(x)
        x, hidden = self.spiral_conv(x, hidden)
        y = self.fc(y)
        y = self.silu(y)
        x = x * y
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class Architecture(StackedHiddenState):
    def __init__(self, depth: int, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__(
            nn.ModuleList([ArchitectureBlock(dim, dim_ff_scale, dropout) for _ in range(depth)]),
            [nn.Parameter(torch.randn(dim, dtype=torch.cfloat)) for _ in range(depth)],
        )
