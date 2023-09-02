import torch
import torch.nn as nn


class SmallConvNet(nn.Module):
    def __init__(self, height: int, width: int, dim_out: int):
        super().__init__()
        self.conv2d1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(
            ((((height - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * ((((width - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * 64,
            dim_out,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.relu(x)
        x = self.conv2d2(x)
        x = self.relu(x)
        x = self.conv2d3(x)
        x = self.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
