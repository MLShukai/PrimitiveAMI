from functools import partial

import torch.nn as nn

from ..forward_dynamics_lit_module import ForwardDynamicsLitModule
from ..inverse_dynamics_lit_module import InverseDynamicsLitModule
from ..ppo_lit_module import PPOLitModule
from .neural_networks import NeuralNetworks


class InverseFowardPPO(NeuralNetworks):
    """Neural network aggregation class for curiosity ppo agent."""

    def __init__(
        self,
        inverse_dynamics: InverseDynamicsLitModule,
        forward_dynamics: partial[ForwardDynamicsLitModule],
        ppo: PPOLitModule,
    ) -> None:
        """Construct this class.

        Args:
            inverse_dynamics (InverseDynamicsLitModule): Instance of InverseDynamicsLitModule.
            forward_dynamics (partial[ForwardDynamicsLitModule]): partial instance of ForwardDynamicsLitModule.
            ppo (PPOLitModule): Instance of InverseDynamicsLitModule.
        """
        super().__init__()

        self.inverse_dynamics = inverse_dynamics
        self.forward_dynamics = forward_dynamics(inverse_dynamics.net.observation_encoder)
        self.ppo = ppo

    def build_agent_models(self) -> dict[str, nn.Module]:
        """Build models for CuriosityPPOAgent."""
        models = {
            "embedding": self.inverse_dynamics.net.observation_encoder,
            "dynamics": self.forward_dynamics.forward_dynamics_net,
            "policy": self.ppo.net,
        }

        return models
