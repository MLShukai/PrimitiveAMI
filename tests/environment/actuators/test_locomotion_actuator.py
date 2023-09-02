import pytest
import torch
from pytest_mock import MockerFixture
from vrchat_io.controller.wrappers.osc import AxesLocomotionWrapper

from src.environment.actuators.locomotion_actuator import LocomotionActuator


class TestLocomotionWrapper:
    @pytest.fixture
    def axes_actuator(self, mocker):
        mock_actuator = mocker.Mock(spec=AxesLocomotionWrapper)
        return mock_actuator

    @pytest.mark.parametrize(
        "action",
        (
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device="cuda"),
        ),
    )
    def test_operate(self, axes_actuator, action):
        action_numpy = action.detach().cpu().numpy()
        mod = LocomotionActuator(axes_actuator)
        mod.operate(action)
        mod.actuator.command.assert_called_with(*action_numpy)
