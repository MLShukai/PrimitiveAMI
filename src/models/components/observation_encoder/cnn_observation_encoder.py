import torch.nn as nn
from torch import Tensor

from src.models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)


class CNNObservationEncoder(ObservationEncoder):
    def __init__(self, dim_embed: int, height: int, width: int):
        super().__init__()
        # CNNのパラメータ疎いので調整してもらえるとよいです
        self.conv2d1 = nn.Conv2d(3, 8, 5)
        self.conv2d2 = nn.Conv2d(8, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear((((height - 4) // 2) - 4) // 2 * (((width - 4) // 2) - 4) // 2 * 16, dim_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2d2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
