import torch
import torch.nn as nn
from torch.distributions import Categorical

from ....utils.model import MultiCategoricals
from .stochastic_policy import StochasticPolicy


class DiscretePolicy(StochasticPolicy):
    """Policy head for discrete action space."""

    def __init__(self, dim_in: int, action_choices_per_category: list[int]) -> None:
        """Constructs policy.

        Args:
            dim_in: Input dimension size of tensor.
            action_choices_per_category: List of action choice count per category.
        """
        super().__init__()

        self.heads = nn.ModuleList()
        for choice in action_choices_per_category:
            self.heads.append(nn.Linear(dim_in, choice, bias=False))

    def forward(self, input: torch.Tensor) -> MultiCategoricals:

        categoricals = []
        for head in self.heads:
            logits = head(input)
            categoricals.append(Categorical(logits=logits))

        return MultiCategoricals(categoricals)
