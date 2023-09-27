from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from ..small_conv_net import SmallConvNet
from ..small_deconv_net import SmallDeconvNet
from .observation_encoder import ObservationEncoder


class Encoder(ObservationEncoder):
    def __init__(self, small_conv_net: SmallConvNet, min_stddev=0.0):
        """VAEのエンコーダのコンストラクタです.

        Args:
            dim_embed (int): 潜在変数ベクトルの次元数
            height (int): 入力画像の高さ
            width (int): 入力画像の横幅
            channels (int, optional): 入力画像のチャネル数。 Defaults to 3.
        """
        super().__init__()
        self.conv_net = small_conv_net
        self.min_stddev = min_stddev

    def encode(self, x: Tensor):
        mu_sigma = self.conv_net(x)
        mu, sigma = torch.chunk(mu_sigma, chunks=2, dim=-1)
        sigma = torch.nn.functional.softplus(sigma) + self.min_stddev
        distribution = Normal(mu, sigma)
        return distribution

    def forward(self, x: Tensor):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(self, small_deconv_net: SmallDeconvNet):
        """VAEのデコーダのコンストラクタです。

        Args:
            dim_embed (int): 潜在変数ベクトルの次元数
            height (int): 入力画像の高さ
            width (int): 入力画像の横幅
            channels (int, optional): 入力画像のチャネル数。 Defaults to 3.
        """
        super().__init__()
        self.deconv_net = small_deconv_net

    def decode(self, z: Tensor):
        rec_img = self.deconv_net(z)
        return rec_img

    def forward(self, z: Tensor):
        return self.decode(z)


class VAE:
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """VAEのコンストラクタです。

        Args:
            encoder (VAEEncoder): VAEのエンコーダ
            decoder (VAEDecoder): VAEのデコーダ
        """
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """encodeとdecodeを行い、潜在変数と再構成画像を返します。

        Args:
            x (Tensor): 入力画像

        Returns:
            tuple[Tensor, Tensor]: 再構成画像と潜在変数の分布オブジェクトのタプル
        """
        z_dist: Normal = self.encoder(x)
        z_sampled = z_dist.rsample()
        x_reconstructed = self.decoder(z_sampled)
        return x_reconstructed, z_dist

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x)
