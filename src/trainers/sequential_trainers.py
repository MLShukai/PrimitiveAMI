from .trainer import Trainer


class SequentialTrainers:
    def __init__(self, *trainers: Trainer) -> None:
        self.trainers = trainers

    def train(self) -> None:
        for trainer in self.trainers:
            trainer.train()
