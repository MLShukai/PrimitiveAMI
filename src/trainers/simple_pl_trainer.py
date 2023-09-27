from functools import partial

import lightning.pytorch as pl
from lightning import LightningModule
from torch.utils.data import DataLoader

from ..data_collectors.data_collector import DataCollector
from .trainer import Trainer


class SimplePLTrainer(Trainer):
    def __init__(
        self,
        module: LightningModule,
        data_collector: DataCollector,
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
        if self.pl_trainer.fit_loop.max_epochs is not None:
            self.pl_trainer.fit_loop.max_epochs += self.pl_trainer.fit_loop.max_epochs
