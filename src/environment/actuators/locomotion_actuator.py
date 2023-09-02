import torch
from vrchat_io.controller.wrappers.osc import AxesLocomotionWrapper

from .actuator import Actuator


class LocomotionActuator(Actuator):
    def __init__(self, actuator: AxesLocomotionWrapper):
        self.actuator = actuator

    def operate(self, action: torch.tensor) -> None:
        _action = action.detach().cpu().numpy()
        self.actuator.command(*_action)
