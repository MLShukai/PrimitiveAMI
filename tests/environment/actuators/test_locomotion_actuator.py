import pytest
import torch
from pytest_mock import MockerFixture
from vrchat_io.controller.wrappers.osc import AxesLocomotionWrapper

from src.environment.actuators.locomotion_actuator import (
    LocomotionActuator,
    get_sleep_action,
)


class TestLocomotionWrapper:
    @pytest.fixture
    def axes_actuator(self, mocker):
        mock_actuator = mocker.Mock(spec=AxesLocomotionWrapper)
        return mock_actuator

    def test__init__(self, axes_actuator):
        mod = LocomotionActuator(axes_actuator)
        assert mod.actuator is axes_actuator

    def _test_operate(self, axes_actuator, action):
        action_numpy = action.detach().cpu().numpy()
        mod = LocomotionActuator(axes_actuator)
        mod.operate(action)
        mod.actuator.command.assert_called_with(*action_numpy)

    def test_operate_cpu(self, axes_actuator):
        action = torch.tensor([0.0, 0.0, 0.0])
        self._test_operate(axes_actuator, action)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda device")
    def test_operate_cuda(self, axes_actuator):
        action = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        self._test_operate(axes_actuator, action)


def test_sleep_action():
    assert torch.equal(get_sleep_action(), torch.zeros(3))
