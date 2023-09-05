import torch
from vrchat_io.controller.wrappers.osc import AxesLocomotionWrapper

from .actuator import Actuator


class LocomotionActuator(Actuator):
    def __init__(self, actuator: AxesLocomotionWrapper):
        self.actuator = actuator

    def operate(self, action: torch.Tensor) -> None:
        _action = action.detach().cpu().numpy().tolist()
        self.actuator.command(_action[0], _action[1], _action[2])
