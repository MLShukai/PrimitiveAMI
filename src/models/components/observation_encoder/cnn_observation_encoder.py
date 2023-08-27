import torch.nn as nn
from torch import Tensor

from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)


class CNNObservationEncoder(ObservationEncoder):
    def __init__(self, dim_embed: int, height: int, width: int):
        super().__init__()
        # CNNのパラメータ疎いので調整してもらえるとよいです
        self.conv2d1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc = nn.Linear(((((height-(8-4))//4-(4-2))//2-(3-1))//1) * ((((width-(8-4))//4-(4-2))//2-(3-1))//1) * 64, dim_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d1(x)
        x = self.leakyrelu(x)
        x = self.conv2d2(x)
        x = self.leakyrelu(x)
        x = self.conv2d3(x)
        x = self.leakyrelu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
