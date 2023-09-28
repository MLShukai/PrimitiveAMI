import torch
import torch.nn as nn

from src.utils.model import SequentialModuleList


def test_SequentialModuleList():
    mod = SequentialModuleList([nn.Linear(2, 2) for _ in range(3)])
    mod(torch.randn(2))
