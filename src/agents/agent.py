from abc import ABC, abstractmethod
from typing import Any, Optional


class Agent(ABC):
    """Abstract Agent class."""

    def wakeup(self, observation: Any) -> Optional[Any]:
        """Setup procedure of Agent.

        Args:
            observation (Any): Initial observation from environment.

        Returns:
            action (Optional[Any]): Initial action to environment on interaction. You can return no action.
        """
        return

    @abstractmethod
    def step(self, observation: Any) -> Any:
        """Process observation, and return action to environment.

        Args:
            observation (Any): Observation from environment.

        Returns:
            action (Any): Action to environment.
        """
        raise NotImplementedError

    def sleep(self, observation: Any) -> Optional[Any]:
        """Sleep procedure of Agent.

        Args:
            observation (Any): Final observation from environment.

        Returns:
            action (Optional[Any]): Final action on interaction. You can return no action.
        """
        return
