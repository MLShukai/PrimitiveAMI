from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class SmallDeconvNet(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        dim_in: int,
        positional_bias: bool = True,
        nl: Callable = nn.LeakyReLU(negative_slope=0.2),
    ):
        """潜在変数から画像の再構成を行います。 (256,
        256)に合わせるため、2回目のConvTranspose2dにoutput_padding=1を追加しています。

        Args:
            height (int): 再構成画像の高さ
            width (int): 再構成画像の横幅
            channels (int): 再構成画像のチャネル数
            dim_in (int): 潜在変数ベクトルの次元数
            positional_bias (bool): 最後にバイアス項を足すかどうか
            nl (Callable): 活性化関数
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.kernel_sizes = ((4, 4), (8, 8), (8, 8))
        self.strides = ((2, 2), (2, 2), (3, 3))
        self.paddings = ((1, 1), (3, 3), (2, 2))

        init_output_size = self.init_output_size
        dim_output_init_fc = init_output_size[0] * init_output_size[1]
        self.fc_init = nn.Linear(dim_in, dim_output_init_fc)

        self.deconv1 = nn.ConvTranspose2d(
            1, 128, kernel_size=self.kernel_sizes[0], stride=self.strides[0], padding=self.paddings[0]
        )
        self.deconv2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=self.kernel_sizes[1],
            stride=self.strides[1],
            padding=self.paddings[1],
            output_padding=1,
        )  # 追加部分
        self.deconv3 = nn.ConvTranspose2d(
            64, 3, kernel_size=self.kernel_sizes[2], stride=self.strides[2], padding=self.paddings[2]
        )
        self.bias = nn.Parameter(torch.zeros(channels, height, width), requires_grad=True) if positional_bias else None
        self.nl = nl

    @property
    def init_output_size(self):
        output_size = (self.height, self.width)
        for kernel_size, stride, padding in zip(self.kernel_sizes[::-1], self.strides[::-1], self.paddings[::-1]):
            output_size = tuple(map(self._compute_input_shape, output_size, kernel_size, stride, padding))
        return output_size

    def _compute_input_shape(
        self,
        edge_output_dim: int,
        kernel_size: int,
        edge_stride: int,
        edge_padding: int,
        dilation: int = 1,
        out_pad: int = 0,
    ):
        return (edge_output_dim - 1 - out_pad - dilation * (kernel_size - 1) + 2 * edge_padding) // edge_stride + 1

    def forward(self, x: Tensor):
        x = self.fc_init(x)
        x = x.view(-1, 1, self.init_output_size[0], self.init_output_size[1])
        x = self.nl(self.deconv1(x))
        x = self.nl(self.deconv2(x))
        x = self.deconv3(x)
        if self.bias is not None:
            x = x + self.bias
        return x
