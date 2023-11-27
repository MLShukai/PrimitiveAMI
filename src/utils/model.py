"""This file contains utility tools for building models."""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution


class SequentialModuleList(nn.ModuleList):
    """Construct dnn modules like `nn.ModuleList`, and forward data like
    `nn.Sequential`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x)
        return x


class MultiDistributions(Distribution):
    """Set of distribution classes."""

    arg_constraints = {}

    def __init__(self, distributions=list[Distribution]) -> None:
        super().__init__(validate_args=False)

        self.dists = distributions

    def sample(self, sample_shape: list[torch.Size] = torch.Size()) -> list[torch.Tensor]:
        return [d.sample(sample_shape) for d in self.dists]

    def log_prob(self, value: list[torch.Tensor]) -> list[torch.Tensor]:
        return [d.log_prob(v) for d, v in zip(self.dists, value)]

    def entropy(self) -> list[torch.Tensor]:
        return [d.entropy() for d in self.dists]


class MultiCategoricals(MultiDistributions):
    """Set of same action torch.Size categorical distributions."""

    def __init__(self, distributions=list[Categorical]) -> None:
        assert len(distributions) > 0
        first_dist = distributions[0]
        assert all(first_dist.batch_shape == d.batch_shape for d in distributions), "All batch shapes must be same."

        super().__init__(distributions)

    def sample(self, sample_shape: list[torch.Size] = torch.Size()) -> torch.Tensor:
        return torch.stack(super().sample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack(super().log_prob(value))

    def entropy(self) -> torch.Tensor:
        return torch.stack(super().entropy())
