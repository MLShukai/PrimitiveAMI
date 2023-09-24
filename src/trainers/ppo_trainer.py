from functools import partial

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..data_collectors.trajectory_data_collector import TrajectoryDataCollector
from ..models.ppo_lit_module import PPOLitModule
from .trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(
        self,
        module: PPOLitModule,
        data_collector: TrajectoryDataCollector,
        dataloader: partial[DataLoader],
        pl_trainer: pl.Trainer,
    ):
        self.module = module
        self.data_collector = data_collector
        self.dataloader = dataloader
        self.pl_trainer = pl_trainer

    def train(self):
        dataset = self.data_collector.get_data()
        dataloader = self.dataloader(dataset=dataset)
        self.pl_trainer.fit(self.module, dataloader)
        self.data_collector.clear()
