from functools import partial

import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset

from ..data_collectors.dynamics_data_collector import DynamicsDataCollector
from ..models.forward_dynamics_lit_module import ForwardDynamicsLitModule
from .trainer import Trainer


class ForwardDynamicsTrainer(Trainer):
    def __init__(
        self,
        module: ForwardDynamicsLitModule,
        data_collector: DynamicsDataCollector,
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
            self.pl_trainer.fit_loop.max_epochs += 1
