from abc import ABC, abstractmethod

from ..agents.agent import Agent
from ..environment.environment import Environment


class Interaction(ABC):
    """Create interaction process between agent and environment."""

    def __init__(self, agent: Agent, environment: Environment):
        """Construct interaction class.

        Args:
            agent (Agent): The agent class that interacts with environment class.
            environment (Environment): The environment class that interacts with agent class.
        """
        self.agent = agent
        self.environment = environment

    @abstractmethod
    def interact(self):
        """Interaction process."""
        raise NotImplementedError
