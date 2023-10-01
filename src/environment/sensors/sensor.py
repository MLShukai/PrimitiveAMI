from abc import ABCMeta, abstractmethod


class Sensor(metaclass=ABCMeta):
    @abstractmethod
    def read(self):
        """Read and return data from the actual sensor."""
        raise NotImplementedError

    def setup(self):
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self):
        """Called at the end of interaction with the agent."""
        pass
