import torch
from vrchat_io.controller.wrappers.osc import (
    AXES_LOCOMOTION_RESET_VALUES,
    AxesLocomotionWrapper,
)

from .actuator import Actuator, ActuatorWrapper


class LocomotionActuator(Actuator):
    def __init__(self, actuator: AxesLocomotionWrapper):
        self.actuator = actuator

    def operate(self, action: torch.Tensor) -> None:
        _action = action.detach().cpu().numpy().tolist()
        self.actuator.command(_action[0], _action[1], _action[2])

    def teardown(self):
        return self.operate(get_sleep_action())


def get_sleep_action() -> torch.Tensor:
    """Return sleep action for LocomotionActuator.

    Returns:
        torch.Tensor: sleep action.
    """
    return torch.tensor(AXES_LOCOMOTION_RESET_VALUES, dtype=torch.float).clone()


class DeadzoneWrapper(ActuatorWrapper):
    """Attach deadzone to action."""

    def __init__(self, actuator: LocomotionActuator, zone_range: float) -> None:
        """Construct actuator wrapper.

        Args:
            actuator: The instance of LocomotionActuator Class.
            zone_range: The range of deadzone. [-zone_range, zone_range] of action will be 0.0
                This value is always >= 0.0 .
        """

        super().__init__(actuator)
        self.zone_range = abs(zone_range)

    def wrap_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip action range with `zone_range`"""
        clipped = action.clone()
        clipped[action.abs() < self.zone_range] = 0.0
        return clipped
