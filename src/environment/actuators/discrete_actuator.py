from typing import Any

import torch
from vrchat_io.controller.osc import RESET_VALUES, Buttons
from vrchat_io.controller.wrappers.osc import MultiInputWrapper

from .actuator import Actuator

ACTION_CHOICES_PER_CATEGORY = [3, 3, 3, 2, 2]

SLEEP_ACTION = torch.tensor(
    [
        0,  # stop to move vertical.
        0,  # stop to move horizontal.
        0,  # stop to turn horizontal.
        0,  # stop to jump.
        0,  # stop to run.
    ],
    dtype=torch.long,
)


class DiscreteActuator(Actuator):
    """Treats discrete actions.

    Input action is 1D integer tensor.

    Actions:
        MoveVertical: [0: stop | 1: forward | 2: backward]
        MoveHorizontal:[0: stop | 1: right | 2: left]
        LookHorizontal: [0: stop | 1: right | 2: left]
        Jump: [0: release | 1: do]
        Run: [0: release | 1: do]
    """

    def __init__(self, controller: MultiInputWrapper) -> None:
        self.controller = controller

    def operate(self, action: torch.Tensor) -> None:
        command = self.convert_action_to_command(action.long().tolist())
        self.controller.command(command)

    def teardown(self) -> None:
        self.controller.command(RESET_VALUES)

    def convert_action_to_command(self, action: list[int]) -> dict[str, Any]:
        """Convert raw action list to command dictionary."""
        command = {}
        command.update(self.move_vertical_to_command(action[0]))
        command.update(self.move_horizontal_to_command(action[1]))
        command.update(self.look_horizontal_to_command(action[2]))
        command.update(self.jump_to_command(action[3]))
        command.update(self.run_to_command(action[4]))
        return command

    @staticmethod
    def move_vertical_to_command(move_vert: int) -> dict[str, int]:
        base_command = {
            Buttons.MoveForward: 0,
            Buttons.MoveBackward: 0,
        }

        match move_vert:
            case 0:  # stop
                return base_command
            case 1:  # forward
                base_command[Buttons.MoveForward] = 1
                return base_command
            case 2:  # backward
                base_command[Buttons.MoveBackward] = 1
                return base_command
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_vert}")

    @staticmethod
    def move_horizontal_to_command(move_horzn: int) -> dict[str, int]:
        base_command = {
            Buttons.MoveLeft: 0,
            Buttons.MoveRight: 0,
        }

        match move_horzn:
            case 0:  # stop
                return base_command
            case 1:  # forward
                base_command[Buttons.MoveLeft] = 1
                return base_command
            case 2:  # backward
                base_command[Buttons.MoveRight] = 1
                return base_command
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_horzn}")

    @staticmethod
    def look_horizontal_to_command(look_horzn: int) -> dict[str, int]:
        base_command = {
            Buttons.LookLeft: 0,
            Buttons.LookRight: 0,
        }

        match look_horzn:
            case 0:  # stop
                return base_command
            case 1:
                base_command[Buttons.LookLeft] = 1
                return base_command
            case 2:
                base_command[Buttons.LookRight] = 1
                return base_command
            case _:
                raise ValueError(f"Choices are 0 to 2! Input: {look_horzn}")

    @staticmethod
    def jump_to_command(jump: int) -> dict[str, int]:
        match jump:
            case 0:
                return {Buttons.Jump: 0}
            case 1:
                return {Buttons.Jump: 1}
            case _:
                raise ValueError(f"Choices are 0 or 1! Input: {jump}")

    @staticmethod
    def run_to_command(run: int) -> dict[str, int]:
        match run:
            case 0:
                return {Buttons.Run: 0}
            case 1:
                return {Buttons.Run: 1}
            case _:
                raise ValueError(f"Choices are 0 or 1! Input: {run}")
