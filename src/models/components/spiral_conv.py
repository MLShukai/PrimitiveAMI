import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * dim_ff_scale, bias=True)
        self.linear_2 = nn.Linear(dim * dim_ff_scale, dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class SpiralConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))  # log(-log(gamma))
        self.phazor = nn.Parameter(torch.exp(2.0j * np.pi * torch.arange(dim) / dim) * torch.abs(torch.randn(dim)))
        self.last_conv = None  # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))  # (dim)
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        len = x.shape[1]
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.expand(batch, self.dim)
        phazor = self.phazor / self.phazor.abs() * torch.exp(-self.phazor.abs())
        phazor_progression = torch.pow(
            phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)
        )  # (len, dim)
        filter = phazor_progression * self.phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len * 2, dim=0)  # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len * 2, dim=1)  # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1, 0, len)  # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.detach().unsqueeze(1) * phazor_progression.unsqueeze(
            0
        ) * phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:, -1, :]

        return conv_with_past.real

    def reset_hidden(self):
        self.last_conv = None

    def set_is_refresh(self, is_refresh: bool):
        self.is_refresh = is_refresh

    def get_hidden(self) -> Tensor:
        return self.last_conv

    def set_hidden(self, hidden: Tensor):
        self.last_conv = hidden


class ArchitectureBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.spiral_conv = SpiralConv(dim)
        self.ffn = FFN(dim, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_ = x
        y = x
        x = self.layer_norm(x)
        x = self.spiral_conv(x)
        y = self.fc(y)
        y = self.silu(y)
        x = x * y
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def set_is_refresh(self, is_refresh: bool):
        self.spiral_conv.set_is_refresh(is_refresh)

    def get_hidden(self) -> Tensor:
        return self.spiral_conv.get_hidden()

    def set_hidden(self, hidden: Tensor):
        self.spiral_conv.set_hidden(hidden)


class Architecture(nn.Module):
    def __init__(self, depth: int, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.block_list = nn.ModuleList([ArchitectureBlock(dim, dim_ff_scale, dropout) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.block_list:
            x = block(x)
        return x

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh: bool):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

    def get_hidden(self) -> list[tuple[Tensor, Tensor]]:
        hidden_list = []
        for block in self.block_list:
            hidden_list.append(block.get_hidden())
        return hidden_list

    def set_hidden(self, hidden_list):
        for i, block in enumerate(self.block_list):
            block.set_hidden(hidden_list[i])
