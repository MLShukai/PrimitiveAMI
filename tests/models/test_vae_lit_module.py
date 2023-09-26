from functools import partial

import pytest
import torch
from torch.optim import Adam

from src.models.components.observation_encoder.vae import VAE, VAEDecoder, VAEEncoder
from src.models.components.small_conv_net import SmallConvNet
from src.models.components.small_deconv_net import SmallDeconvNet
from src.models.vae_lit_module import VAELitModule as cls


class TestVAELitModule:
    batch_size = 8
    in_shape = (batch_size, 3, 256, 256)
    out_shape = (batch_size, 256)

    @pytest.fixture
    def batch(self):
        batch = torch.randn(self.in_shape, requires_grad=True)
        return batch

    @pytest.fixture
    def encoder(self):
        conv_net = SmallConvNet(self.in_shape[2], self.in_shape[3], self.in_shape[1], 2 * self.out_shape[-1])
        return VAEEncoder(conv_net)

    @pytest.fixture
    def decoder(self):
        deconv_net = SmallDeconvNet(self.in_shape[2], self.in_shape[3], self.in_shape[1], self.out_shape[-1])
        return VAEDecoder(deconv_net)

    @pytest.fixture
    def vae_net(self, encoder, decoder):
        vae_net = VAE(encoder, decoder)
        return vae_net

    @pytest.fixture
    def optimizer(self):
        return partial(Adam)

    @pytest.fixture
    def lit_module(self, vae_net):
        optimizer = partial(Adam)
        lit_module = cls(vae_net, optimizer)
        return lit_module

    def test__init__(self, vae_net, optimizer):
        mod = cls(vae_net, optimizer)
        assert mod.net is vae_net
        assert mod.hparams.optimizer is optimizer

    def test_training_step(self, batch, lit_module):
        loss = lit_module.training_step(batch, 0)
        assert type(loss) == torch.Tensor
