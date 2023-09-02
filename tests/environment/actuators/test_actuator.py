from src.environment.actuators.actuator import Actuator


class TestActuator:
    def test_is_abstract_method(self):
        Actuator.__abstractmethods__ == frozenset({"operate"})
