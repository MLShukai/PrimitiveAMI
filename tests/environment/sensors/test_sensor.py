from src.environment.sensors.sensor import Sensor


class TestSensor:
    def test_is_abstract(self):
        assert Sensor.__abstractmethods__ == frozenset({"read"})
