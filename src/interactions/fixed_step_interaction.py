from ..agents.agent import Agent
from ..environment.environment import Environment
from .interaction import Interaction


class FixedStepInteraction(Interaction):
    """Interact with environment for fixed number of steps."""

    def __init__(self, agent: Agent, environment: Environment, num_steps: int):
        """Construct fixed step interaction class.

        Args:
            agent (Agent): The agent class that interacts with environment class.
            environment (Environment): The environment class that interacts with agent class.
            num_steps (int): The number of steps to interact.
        """

        super().__init__(agent, environment)
        self.num_steps = num_steps

    def initialize(self):
        """Initialize interaction process."""
        self.environment.setup()
        initial_action = self.agent.wakeup(self.environment.observe())
        if initial_action is not None:
            self.environment.affect(initial_action)

    def finalize(self):
        """Finalize interaction process."""
        final_action = self.agent.sleep(self.environment.observe())
        if final_action is not None:
            self.environment.affect(final_action)
        self.environment.teardown()

    def mainloop(self):
        """Interact with environment for num_steps."""
        for _ in range(self.num_steps):
            obs = self.environment.observe()
            action = self.agent.step(obs)
            self.environment.affect(action)

    def interact(self):
        """Interaction process."""
        self.initialize()
        self.mainloop()
        self.finalize()
