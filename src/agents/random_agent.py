from typing import Any

import torch

from .agent import Agent


class RandomAgent(Agent):
    """Create action randomly.

    The action is sampled from normal distribution.
    """

    def __init__(
        self, action_size: int, mean: float = 0.0, std: float = 1.0, dtype: torch.dtype = torch.float32
    ) -> None:
        """Initialize random agent.

        Args:
            action_size (int): The vector length of action.
            mean (float): The mean of normal distribution.
            std (float): The std of normal distribution.
            dtype (torch.dtype): Data type of tensor.
        """
        self.action_size = action_size
        self.mean = mean
        self.std = std
        self.dtype = dtype

    def step(self, observation: Any) -> torch.Tensor:
        """Return random action."""
        return torch.tanh(torch.randn(self.action_size, dtype=self.dtype) * self.std + self.mean)
