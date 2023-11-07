from functools import partial
from typing import Type

import torch.nn as nn

from ..components.observation_encoder.observation_encoder import ObservationEncoder
from ..components.observation_encoder.vae import DeterministicEncoderWrapper
from ..forward_dynamics_lit_module import ForwardDynamicsLitModule
from ..ppo_lit_module import PPOLitModule
from ..vae_lit_module import VAELitModule
from .neural_networks import NeuralNetworks


class InverseForwardPPO(NeuralNetworks):
    """Neural network aggregation class for curiosity ppo agent."""

    def __init__(
        self,
        vae: VAELitModule,
        forward_dynamics: partial[ForwardDynamicsLitModule],
        ppo: PPOLitModule,
        encoder_wrapper_cls: Type[ObservationEncoder] = DeterministicEncoderWrapper,
    ) -> None:
        """Construct this class.

        Args:
            vae: Instance of VAELitModule.
            forward_dynamics: partial instance of ForwardDynamicsLitModule.
            ppo: Instance of InverseDynamicsLitModule.
            encoder_wrapper_cls: The wrapper for converting vae encoder output distribution to Tensor.
        """
        super().__init__()

        self.vae = vae
        self.encoder_wrapper_cls = encoder_wrapper_cls
        self.forward_dynamics = forward_dynamics(obs_encoder=encoder_wrapper_cls(self.vae.net.encoder))
        self.ppo = ppo

    def build_agent_models(self) -> dict[str, nn.Module]:
        """Build models for CuriosityPPOAgent."""
        models = {
            "embedding": self.encoder_wrapper_cls(self.vae.net.encoder),
            "dynamics": self.forward_dynamics.forward_dynamics_net,
            "policy": self.ppo.net,
        }

        return models
