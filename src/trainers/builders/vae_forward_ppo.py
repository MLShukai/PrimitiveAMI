from functools import partial

from ...data_collectors.aggregations.dict_data_collectors import DictDataCollectors
from ...models.aggregations import vae_forward_ppo as neuralnets
from ..forward_dynamics_trainer import ForwardDynamicsTrainer
from ..ppo_trainer import PPOTrainer
from ..sequential_trainers import SequentialTrainers
from ..simple_pl_trainer import SimplePLTrainer
from .trainers_builder import TrainersBuilder


class VAEForwardPPO(TrainersBuilder):
    """Aggregates VAE, Forward, PPO Trainers and build them."""

    def __init__(
        self,
        vae_trainer: partial[SimplePLTrainer],
        forward_dynamics_trainer: partial[ForwardDynamicsTrainer],
        ppo_trainer: partial[PPOTrainer],
    ) -> None:
        """Construct his class.

        Args:
            vae_trainer: Partial instance that module and data_collector are not provided.
            forward_trainer: Partial instance that module and data_collector are not provided.
            ppo_trainer: Partial instance that module and data_collector are not provided.
        """

        self.vae_trainer = vae_trainer
        self.forward_dynamics_trainer = forward_dynamics_trainer
        self.ppo_trainer = ppo_trainer

    def build(self, nets: neuralnets.VAEForwardPPO, data_collectors: DictDataCollectors) -> SequentialTrainers:
        return SequentialTrainers(
            self.vae_trainer(nets.vae, data_collectors["observation"]),
            self.forward_dynamics_trainer(nets.forward_dynamics, data_collectors["dynamics"]),
            self.ppo_trainer(nets.ppo, data_collectors["trajectory"]),
        )
