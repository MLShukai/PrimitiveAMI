import torch
import torch.nn as nn
from torch import Tensor

from .value import Value


class FullyConnectValue(Value):
    def __init__(self, dim_input: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, 1)

    def forward(self, input: Tensor):
        value = self.fc(input)
        return value
