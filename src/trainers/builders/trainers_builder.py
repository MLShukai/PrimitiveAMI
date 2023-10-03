from abc import ABC, abstractmethod

from ...data_collectors.aggregations.data_collectors_aggregation import (
    DataCollectorsAggregation,
)
from ...models.aggregations.neural_networks import NeuralNetworks
from ..trainer import Trainer


class TrainersBuilder(ABC):
    """Abstract interface class for building trainers."""

    @abstractmethod
    def build(self, nets: NeuralNetworks, data_collectors: DataCollectorsAggregation) -> Trainer:
        """Build trainers with neural networks and data collectors.

        Args:
            nets (NeuralNetworks): Aggregation class of NeuralNetwork models.
            data_collectors (DataCollectorsAggregation): data_collector(s) class for providing dataset to each trainers.

        Returns:
            Trainer: Built single trainer class. (Aggregated other trainers.)
        """
        raise NotImplementedError
