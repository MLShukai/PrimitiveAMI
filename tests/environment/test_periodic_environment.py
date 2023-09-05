import pytest
import torch

from src.environment.actuators.locomotion_actuator import LocomotionActuator
from src.environment.interval_adjustors.sleep_interval_adjustor import (
    SleepIntervalAdjustor,
)
from src.environment.periodic_environment import PeriodicEnvironment as cls
from src.environment.sensors.frame_sensor import FrameSensor


class TestPeriodicEnvironment:
    @pytest.fixture
    def sensor(self, mocker):
        mock_sensor = mocker.Mock(spec=FrameSensor)
        mock_sensor.read.return_value = torch.zeros((3, 640, 480))
        return mock_sensor

    @pytest.fixture
    def actuator(self, mocker):
        mock_actuator = mocker.Mock(spec=LocomotionActuator)
        return mock_actuator

    @pytest.fixture
    def adjustor(self, mocker):
        mock_adjustor = mocker.Mock(spec=SleepIntervalAdjustor)
        mock_adjustor.reset.return_value = 0.0
        return mock_adjustor

    @pytest.fixture
    def mod(self, sensor, actuator, adjustor):
        return cls(sensor, actuator, adjustor)

    def test__init__(self, sensor, actuator, adjustor):
        mod = cls(sensor, actuator, adjustor)
        assert mod.sensor is sensor
        assert mod.actuator is actuator
        assert mod.adjustor is adjustor

    def test_setup(self, mocker, sensor, actuator, adjustor):
        mod = mocker.Mock(spec=cls)
        mod.setup()
        assert mod.setup.call_count == 1

    def test_observe(self, mod):
        torch.testing.assert_close(mod.observe(), torch.zeros((3, 640, 480)))

    def test_affect(self, mod):
        action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        mod.affect(action)
        mod.actuator.operate.assert_called_with(action)

    def test_reset_interval_start_time(self, mod):
        assert mod.reset_interval_start_time() == 0.0
        mod.adjustor.reset.assert_called_once()

    def test_adjust_interval(self, mod):
        mod.adjust_interval()
        assert mod.adjustor.adjust.call_count == 1
