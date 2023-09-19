from typing import Optional

import torch
import torch.nn as nn


class SmallConvNet(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        dim_out: int,
        do_batchnorm: bool = False,
        do_layernorm: bool = False,
        nl: Optional[nn.Module] = nn.LeakyReLU(),
        last_nl: Optional[nn.Module] = None,
    ) -> None:
        """Construct small conv net.

        Args:
            height (int): height of pictured frame.
            width (int): width of pictured frame.
            channels (int): channels of pictured frame.
            dim_out (int): The number of dimensions of the output tensor.
            do_batchnorm(bool, optional): Whether to do batchnorm. Defaults to False. https://github.com/openai/large-scale-curiosity/blob/master/utils.py#L133
            do_layernorm (bool, optional): Whether to do layernorm. Defaults to False. https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L7
            nl (Optional[nn.Module], optional): NonLinear function for activation. Defaults to nn.LeakyReLU(). https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L39
            last_nl (Optional[nn.Module], optional): NonLinearFunction for activation for the last layer. Defaults to None. https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L46
        """
        super().__init__()
        self.conv2d1 = nn.Conv2d(channels, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(
            ((((height - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * ((((width - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * 64,
            dim_out,
        )
        self.nl = nl
        self.last_nl = last_nl
        self.do_batchnorm = do_batchnorm
        self.do_layernorm = do_layernorm

    def forward(self, x):
        do_bn = self.do_batchnorm
        x = self.bn1(self.conv2d1(x)) if do_bn else self.conv2d1(x)
        x = self.nl(x)
        x = self.bn2(self.conv2d2(x)) if do_bn else self.conv2d2(x)
        x = self.nl(x)
        x = self.bn3(self.conv2d3(x)) if do_bn else self.conv2d3(x)
        x = self.nl(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        if self.last_nl is not None:
            x = self.last_nl(x)
        if self.do_layernorm:
            layernorm = nn.LayerNorm(x.shape[-1])
            x = layernorm(x)
        return x
