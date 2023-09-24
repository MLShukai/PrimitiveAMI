import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: nn.Module = None):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
