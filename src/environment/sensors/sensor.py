from abc import ABCMeta, abstractmethod


class Sensor(metaclass=ABCMeta):
    @abstractmethod
    def read(self):
        """Read and return data from the actual sensor."""
        raise NotImplementedError
