import torch.nn as nn
from torch import Tensor


class HiddenState(nn.Module):
    def __init__(self, module: nn.Module, hidden_init: Tensor):
        super().__init__()
        self.module = module
        self.hidden = None
        self.hidden_init = hidden_init

    def forward(self, x: Tensor):
        if self.hidden is None:
            self.hidden = self.hidden_init.unsqueeze(0).expand(x.shape[0], -1)

        x, hidden = self.module(x, self.hidden)
        self.hidden = hidden
        return x

    def reset_hidden(self):
        self.hidden = None

    def get_hidden(self):
        return self.hidden

    def set_hidden(self, hidden: Tensor):
        self.last_conv = hidden


class StackedHiddenState(nn.Module):
    def __init__(self, module_list: nn.ModuleList, hidden_init_list: [Tensor]):
        super().__init__()
        self.module_list = nn.ModuleList(
            [HiddenState(module_list[i], hidden_init_list[i]) for i in range(len(module_list))]
        )

    def forward(self, x: Tensor):
        for module in self.module_list:
            x = module(x)
        return x

    def reset_hidden(self):
        for module in self.module_list:
            module.reset_hidden()

    def get_hidden(self):
        return [module.get_hidden() for module in self.module_list]

    def set_hidden(self, hidden_list: [Tensor]):
        for i, module in enumerate(self.module_list):
            module.set_hidden(hidden_list[i])
