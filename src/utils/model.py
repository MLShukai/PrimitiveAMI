"""This file contains utility tools for building models."""
import torch
import torch.nn as nn


class SequentialModuleList(nn.ModuleList):
    """Construct dnn modules like `nn.ModuleList`, and forward data like
    `nn.Sequential`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x)
        return x
