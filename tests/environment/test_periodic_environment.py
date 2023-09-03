from src.environment.periodic_environment import PeriodicEnvironment as cls
import pytest
from src.environment.sensors.frame_sensor import FrameSensor
from src.environment.actuators.locomotion_actuator import LocomotionActuator
from src.environment.interval_adjustors.sleep_interval_adjustor import SleepIntervalAdjustor
import time

adjustor_params = {"interval": 0.1, "offset": 0.}


class TestPeriodicEnvironment:

    @pytest.fixture
    def sensor(self, mocker):
        mock_sensor = mocker.Mock(spec=FrameSensor)
        return mock_sensor
    
    @pytest.fixture
    def actuator(self, mocker):
        mock_actuator = mocker.Mock(spec=LocomotionActuator)
        return mock_actuator
    
    @pytest.fixture
    def adjustor(self):
        adjustor = SleepIntervalAdjustor(adjustor_params["interval"], adjustor_params["offset"])
        return adjustor
    
    def test__init__(self, sensor, actuator, adjustor):
        mod = cls(sensor, actuator, adjustor)
        assert mod.sensor is sensor
        assert mod.actuator is actuator
        assert mod.adjustor is adjustor

    def test_setup(self, mocker):
        with mocker.patch("")
