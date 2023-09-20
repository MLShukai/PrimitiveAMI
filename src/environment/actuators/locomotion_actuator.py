import torch
from vrchat_io.controller.wrappers.osc import (
    AXES_LOCOMOTION_RESET_VALUES,
    AxesLocomotionWrapper,
)

from .actuator import Actuator


class LocomotionActuator(Actuator):
    def __init__(self, actuator: AxesLocomotionWrapper):
        self.actuator = actuator

    def operate(self, action: torch.Tensor) -> None:
        _action = action.detach().cpu().numpy().tolist()
        self.actuator.command(_action[0], _action[1], _action[2])


def get_sleep_action() -> torch.Tensor:
    """Return sleep action for LocomotionActuator.

    Retruns:
        torch.Tensor: sleep action.
    """
    return torch.tensor(AXES_LOCOMOTION_RESET_VALUES, dtype=torch.float).clone()
