import torch.nn as nn
from torch import Tensor

from ..small_conv_net import SmallConvNet
from .observation_encoder import ObservationEncoder


class CNNObservationEncoder(ObservationEncoder):
    def __init__(self, dim_embed: int, height: int, width: int):
        super().__init__()
        self.conv_net = SmallConvNet(height, width, dim_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_net(x)
        return x
