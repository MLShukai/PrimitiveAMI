from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """Abstract class for Environment."""

    @abstractmethod
    def observe(self):
        """Return sensor observations."""
        raise NotImplementedError

    @abstractmethod
    def affect(self, action):
        """Apply actions to the Environment."""
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        """Called at the start of interaction with the agent."""
        raise NotImplementedError

    @abstractmethod
    def teardown(self):
        """Called at the end of interaction with the agent."""
        raise NotImplementedError

    def step(self, action):
        """Execute affect and observe in this order."""
        self.affect(action)
        return self.observe()
