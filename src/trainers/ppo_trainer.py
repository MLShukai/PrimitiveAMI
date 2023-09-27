from .simple_pl_trainer import SimplePLTrainer


class PPOTrainer(SimplePLTrainer):
    def train(self):
        super().train()
        self.data_collector.clear()
