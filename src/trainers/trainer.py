from abc import ABC, abstractmethod


class Trainer(ABC):
    """Abstract class for trainers."""

    @abstractmethod
    def train(self):
        """Train the model."""
        raise NotImplementedError
