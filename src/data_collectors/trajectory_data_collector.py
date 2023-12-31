from collections import deque
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ..utils.step_record import RecordKeys as RK
from .data_collector import DataCollector


class TrajectoryDataCollector(DataCollector):
    """Collecting trajectory data for ppo training.

    Returning objects are;
        - observations
        - actions
        - action log probabilities
        - advantages
        - returns
        - values
    """

    def __init__(
        self,
        max_size: int,
        gamma: float = 0.99,  # https://github.com/openai/large-scale-curiosity/blob/e0a698676d19307a095cd4ac1991c4e4e70e56fb/run.py#L174
        gae_lambda: float = 0.95,  # https://github.com/openai/large-scale-curiosity/blob/e0a698676d19307a095cd4ac1991c4e4e70e56fb/run.py#L173
    ) -> None:
        """Construct this class.

        Args:
            max_size (int): The max size of internal buffer.
            gamma (float): Discount factor.
            gae_lambda (float): The lambda of generalized advantage estimation.
        """
        self.max_size = max_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = deque(maxlen=max_size)  # If reached maxlen, auto popping left.
        self.actions = deque(maxlen=max_size)
        self.logprobs = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.final_next_value: Tensor = torch.tensor([0.0])

    def collect(self, step_record: dict[str, Tensor]) -> None:
        """Collect data from step record."""
        self.observations.append(step_record[RK.OBSERVATION].cpu().clone())
        self.actions.append(step_record[RK.ACTION].cpu().clone())
        self.logprobs.append(step_record[RK.ACTION_LOG_PROBABILITY].cpu().clone())
        self.rewards.append(step_record[RK.REWARD].cpu().clone().flatten())
        self.values.append(step_record[RK.VALUE].cpu().clone().flatten())
        self.final_next_value = step_record[RK.NEXT_VALUE].cpu().clone().flatten()

    def get_data(self) -> TensorDataset:
        """Provide dataset for training."""
        observations = torch.stack(list(self.observations))
        actions = torch.stack(list(self.actions))
        logprobs = torch.stack(list(self.logprobs))
        rewards = torch.stack(list(self.rewards))
        values = torch.stack(list(self.values))
        advantages = compute_advantage(rewards, values, self.final_next_value, self.gamma, self.gae_lambda)
        returns = advantages + values

        dataset = TensorDataset(
            observations,
            actions,
            logprobs,
            advantages,
            returns,
            values,
        )

        return dataset

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.final_next_value = torch.tensor([0.0])

    def state_dict(self) -> dict[str, Any]:
        state = {
            "observations": self.observations,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "values": self.values,
            "final_next_value": self.final_next_value,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.observations = deque(state_dict["observations"], maxlen=self.max_size)
        self.actions = deque(state_dict["actions"], maxlen=self.max_size)
        self.logprobs = deque(state_dict["logprobs"], maxlen=self.max_size)
        self.rewards = deque(state_dict["rewards"], maxlen=self.max_size)
        self.values = deque(state_dict["values"], maxlen=self.max_size)
        self.final_next_value = state_dict["final_next_value"]


def compute_advantage(
    rewards: Tensor, values: Tensor, final_next_value: Tensor, gamma: float, gae_lambda: float
) -> Tensor:
    """Compute advantages from values.

    Args:
        rewards (Tensor): shape (step length, )
        values (Tensor): shape (step length, )
        final_next_value (Tensor): shape (1,)
        gamma (float): Discount factor.
        gae_lambda (float): The lambda of generalized advantage estimation.

    Returns:
        advantages (Tensor): shape
    """
    advantages = torch.empty_like(values)

    lastgaelam = 0.0

    for t in reversed(range(values.size(0))):
        if t == values.size(0) - 1:
            nextvalues = final_next_value
        else:
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam

    return advantages
