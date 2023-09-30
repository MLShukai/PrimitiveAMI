from ..data_collectors.trajectory_data_collector import TrajectoryDataCollector
from .simple_pl_trainer import SimplePLTrainer


class PPOTrainer(SimplePLTrainer):

    data_collector: TrajectoryDataCollector

    def train(self):
        super().train()
        self.data_collector.clear()
