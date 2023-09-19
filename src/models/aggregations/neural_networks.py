from abc import ABCMeta, abstractmethod

import torch.nn as nn


class NeuralNetworks(nn.Module, metaclass=ABCMeta):
    """Aggregates neural network models (modules) and provide agent models."""

    @abstractmethod
    def build_agent_models(self) -> dict[str, nn.Module]:
        """Build models for constructing agent module.

        Returns:
            dict[str, nn.Module]: Agent's model keyword args
        """
        raise NotImplementedError
