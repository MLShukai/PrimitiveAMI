from functools import partial

from ...data_collectors.aggregations.dict_data_collectors import DictDataCollectors
from ...models.aggregations import inverse_forward_ppo as neuralnets
from ..forward_dynamics_trainer import ForwardDynamicsTrainer
from ..inverse_dynamics_trainer import InverseDynamicsTrainer
from ..ppo_trainer import PPOTrainer
from ..sequential_trainers import SequentialTrainers
from .trainers_builder import TrainersBuilder


class InverseForwardPPO(TrainersBuilder):
    """Aggregates Inverse, Forward, PPO Trainers and build them."""

    def __init__(
        self,
        inverse_dynamics: partial[InverseDynamicsTrainer],
        forward_dynamics: partial[ForwardDynamicsTrainer],
        ppo: partial[PPOTrainer],
    ) -> None:
        """Construct this class.

        Args:
            inverse_dynamics (partial[InverseDynamicsTrainer]): Partial instance that module and data_collector are not provided.
            forward_dynamics (partial[ForwardDynamicsTrainer]): Partial instance that module and data_collector are not provided.
            ppo (partial[PPOTrainer]): Partial instance that module and data_collector are not provided.
        """

        self.inverse_dynamics = inverse_dynamics
        self.forward_dynamics = forward_dynamics
        self.ppo = ppo

    def build(self, nets: neuralnets.InverseForwardPPO, data_collectors: DictDataCollectors) -> SequentialTrainers:
        return SequentialTrainers(
            self.inverse_dynamics(nets.inverse_dynamics, data_collectors["dynamics"]),
            self.forward_dynamics(nets.forward_dynamics, data_collectors["dynamics"]),
            self.ppo(nets.ppo, data_collectors["trajectory"]),
        )
