import pytest
import torch
from pytest_mock import MockerFixture

from src.environment.actuators.discrete_actuator import (
    Buttons,
    DiscreteActuator,
    MultiInputWrapper,
)


class TestDiscreteActuator:
    @pytest.fixture
    def mock_controller(self, mocker: MockerFixture):
        return mocker.Mock(MultiInputWrapper)

    @pytest.fixture
    def actuator(self, mock_controller):
        return DiscreteActuator(mock_controller)

    @pytest.mark.parametrize(
        "action, expected_command",
        [
            (
                [0, 0, 0, 0, 0],
                {
                    Buttons.MoveForward: 0,
                    Buttons.MoveBackward: 0,
                    Buttons.MoveLeft: 0,
                    Buttons.MoveRight: 0,
                    Buttons.LookLeft: 0,
                    Buttons.LookRight: 0,
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
            (
                [1, 1, 1, 1, 1],
                {
                    Buttons.MoveForward: 1,
                    Buttons.MoveBackward: 0,
                    Buttons.MoveLeft: 1,
                    Buttons.MoveRight: 0,
                    Buttons.LookLeft: 1,
                    Buttons.LookRight: 0,
                    Buttons.Jump: 1,
                    Buttons.Run: 1,
                },
            ),
            (
                [2, 2, 2, 0, 0],
                {
                    Buttons.MoveForward: 0,
                    Buttons.MoveBackward: 1,
                    Buttons.MoveLeft: 0,
                    Buttons.MoveRight: 1,
                    Buttons.LookLeft: 0,
                    Buttons.LookRight: 1,
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
        ],
    )
    def test_convert_action_to_command(self, actuator, action, expected_command):
        assert actuator.convert_action_to_command(action) == expected_command

    def test_operate(self, actuator: DiscreteActuator):

        actuator.operate(torch.tensor([0, 0, 0, 0, 0]))

        with pytest.raises(ValueError):
            actuator.operate(torch.tensor([3, 3, 3, 2, 2]))

    def test_teardown(self, actuator):
        actuator.teardown()
