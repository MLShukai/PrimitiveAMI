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
        self.phazor = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))  # log(-log(gamma))
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))  # log(-log(gamma))
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

    def randomize_init(self):
        self.last_conv_init.value = torch.randn(self.dim, dtype=torch.cfloat)

    def set_is_refresh(self, is_refresh: bool):
        self.is_refresh = is_refresh

    def get_hidden(self) -> Tensor:
        return self.last_conv

    def set_hidden(self, hidden: Tensor):
        self.last_conv = hidden


class ArchitectureBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_scale: float, dropout: float):
        super().__init__()
        self.spiral_conv_1 = SpiralConv(dim)
        self.spiral_conv_2 = SpiralConv(dim)
        self.ffn = FFN(dim, dim_ff_scale, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_ = x
        x = self.layer_norm(x)
        x = self.spiral_conv_1(x)
        x = self.silu(x)
        x = self.spiral_conv_2(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv_1.reset_hidden()
        self.spiral_conv_2.reset_hidden()

    def randomize_init(self):
        self.spiral_conv_1.randomize_init()
        self.spiral_conv_2.randomize_init()

    def set_is_refresh(self, is_refresh: bool):
        self.spiral_conv_1.set_is_refresh(is_refresh)
        self.spiral_conv_2.set_is_refresh(is_refresh)

    def get_hidden(self) -> tuple[Tensor, Tensor]:
        return (self.spiral_conv_1.get_hidden(), self.spiral_conv_2.get_hidden())

    def set_hidden(self, hidden: tuple[Tensor, Tensor]):
        hidden_1, hidden_2 = hidden
        self.spiral_conv_1.set_hidden(hidden_1)
        self.spiral_conv_2.set_hidden(hidden_2)


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

    def randomize_init(self):
        for block in self.block_list:
            block.randomize_init()

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
