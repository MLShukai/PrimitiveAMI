from functools import partial

import pytest
import torch

from src.models.components.observation_encoder.vae import VAE, VAEDecoder, VAEEncoder
from src.models.components.small_conv_net import SmallConvNet
from src.models.components.small_deconv_net import SmallDeconvNet


class TestVAEs:
    HEIGHT = 256
    WIDTH = 256
    CHANNELS = 3
    DIM_EMBED = 256


class TestVAEEncoder(TestVAEs):
    @pytest.fixture
    def small_conv_net(self):
        net = SmallConvNet(self.HEIGHT, self.WIDTH, self.CHANNELS, 2 * self.DIM_EMBED)
        return net

    def test__init__(self, small_conv_net):
        mod = VAEEncoder(small_conv_net)
        assert mod.conv_net is small_conv_net

    def test_encode(self, small_conv_net):
        mod = VAEEncoder(small_conv_net)
        x = torch.randn(8, self.CHANNELS, self.WIDTH, self.HEIGHT)
        z_dist = mod.encode(x)
        assert isinstance(z_dist, torch.distributions.Normal)

    def test_forward(self, small_conv_net):
        mod = VAEEncoder(small_conv_net)
        x = torch.randn(8, self.CHANNELS, self.WIDTH, self.HEIGHT)
        z_dist = mod(x)
        assert isinstance(z_dist, torch.distributions.Normal)


class TestVAEDecode(TestVAEs):
    @pytest.fixture
    def small_deconv_net(self):
        net = SmallDeconvNet(self.HEIGHT, self.WIDTH, self.CHANNELS, self.DIM_EMBED)
        return net

    def test__init__(self, small_deconv_net):
        mod = VAEDecoder(small_deconv_net)
        assert mod.deconv_net is small_deconv_net

    def test_decode(self, small_deconv_net):
        mod = VAEDecoder(small_deconv_net)
        z = torch.randn(8, self.DIM_EMBED)
        rec_x = mod.decode(z)
        assert rec_x.shape == torch.Size((8, self.CHANNELS, self.HEIGHT, self.WIDTH))


class TestVAE(TestVAEs):
    @pytest.fixture
    def encoder(self):
        encoder = VAEEncoder(SmallConvNet(self.HEIGHT, self.WIDTH, self.CHANNELS, 2 * self.DIM_EMBED))
        return encoder

    @pytest.fixture
    def decoder(self):
        decoder = VAEDecoder(SmallDeconvNet(self.HEIGHT, self.WIDTH, self.CHANNELS, self.DIM_EMBED))
        return decoder

    def test__init__(self, encoder, decoder):
        mod = VAE(encoder, decoder)
        assert mod.encoder is encoder
        assert mod.decoder is decoder

    def test__forward(self, encoder, decoder):
        mod = VAE(encoder, decoder)
        x = torch.randn(8, self.CHANNELS, self.HEIGHT, self.WIDTH)
        rec_x, z = mod(x)
        assert z.shape == torch.Size((8, self.DIM_EMBED))
