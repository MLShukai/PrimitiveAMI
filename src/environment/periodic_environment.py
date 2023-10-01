import torch

from .actuators.actuator import Actuator
from .environment import Environment as Environment
from .interval_adjustors.interval_adjustor import IntervalAdjustor
from .sensors.sensor import Sensor


class PeriodicEnvironment(Environment):
    def __init__(self, sensor: Sensor, actuator: Actuator, adjustor: IntervalAdjustor):
        self.sensor = sensor
        self.actuator = actuator
        self.adjustor = adjustor

    def setup(self):
        self.sensor.setup()
        self.actuator.setup()
        self.reset_interval_start_time()

    def observe(self) -> torch.Tensor:
        return self.sensor.read()

    def affect(self, action: torch.Tensor) -> None:
        self.actuator.operate(action)
        self.adjust_interval()

    def reset_interval_start_time(self) -> float:
        return self.adjustor.reset()

    def adjust_interval(self) -> float:
        return self.adjustor.adjust()

    def teardown(self) -> None:
        self.sensor.teardown()
        self.actuator.teardown()
        
