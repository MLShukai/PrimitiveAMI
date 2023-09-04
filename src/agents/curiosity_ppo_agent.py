from typing import TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..data_collectors.data_collector import DataCollector
from ..models.components.forward_dynamics.forward_dynamics import ForwardDynamics
from ..models.components.observation_encoder.observation_encoder import (
    ObservationEncoder,
)
from ..models.components.policy_value_common_net import PolicyValueCommonNet
from ..models.components.reward.curiosity_reward import CuriosityReward
from ..utils.step_record import RecordKeys as RK
from ..utils.step_record import StepRecord
from .agent import Agent

_tensor_or_module_t = TypeVar("_tensor_or_module_t", Tensor, nn.Module)


class CuriosityPPOAgent(Agent):
    """Curiosity based PPO Agent."""

    def __init__(
        self,
        embedding: ObservationEncoder,
        dynamics: ForwardDynamics,
        policy: PolicyValueCommonNet,
        reward: CuriosityReward,
        data_collector: DataCollector,
        sleep_action: Tensor,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Construct CuriosityPPOAgent class.

        Args:
            embedding (ObservationEncoder): Observation encoder.
            dynamics (ForwardDynamics): Forward dynamics model.
            policy (PolicyValueCommonNet): Policy model.
            reward (CuriosityReward): Curiosity reward model.
            data_collector (DataCollector): Data collector.
            sleep_action (Tensor): Action for entering sleep mode.
            device (Union[str, torch.device]): Device to use.
            dtype (torch.dtype): Data type to use.
        """
        super().__init__()

        self.embedding = embedding
        self.dynamics = dynamics
        self.policy = policy
        self.reward = reward
        self.data_collector = data_collector
        self.sleep_action = sleep_action
        self.device = device
        self.dtype = dtype

        self.step_record = StepRecord()

    def wakeup(self, observation: Tensor) -> Tensor:
        """Wakeup process of agent."""
        # t = 0
        self._setup_models_and_buffers()

        observation = self._to_device_dtype(observation)
        embed_obs = self._embed_observation(observation)
        action, action_log_prob, value = self._take_actions_and_value(observation)
        pred_next_embed_obs = self._predict_next_embed_observation(self.sleep_action, embed_obs, action)

        # Store data into step record
        self._clear_step_record()
        self._store_current_step_data(
            previous_action=self.sleep_action,
            observation=observation,
            embed_obs=embed_obs,
            action=action,
            action_log_prob=action_log_prob,
            value=value,
            predicted_next_embed_obs=pred_next_embed_obs,
        )

        return action

    def step(self, observation: Tensor) -> Tensor:
        """Step interaction process of agent."""
        # t = 1, 2, 3, ...
        # ----------- Observation is `NEXT` now. ----------- #
        observation = self._to_device_dtype(observation)

        # Compute reward
        pred_next_embed_obs = self.step_record[RK.PREDICTED_NEXT_EMBED_OBSERVATION]
        embed_obs = self._embed_observation(observation)
        reward = self._compute_reward(pred_next_embed_obs, embed_obs)

        # Store data into step record
        self._store_next_step_data(observation, embed_obs, reward)

        # Data collection
        self.data_collector.collect(self.step_record.copy())

        # ---- Observation is `CURRENT` now. (Move to next step) ---- #
        # Take action
        action, action_log_prob, value = self._take_actions_and_value(observation)

        # Predict next embedded observation
        prev_action = self.step_record[RK.ACTION]
        pred_next_embed_obs = self._predict_next_embed_observation(prev_action, embed_obs, action)

        # Store data into step record for next step
        self._store_current_step_data(
            previous_action=prev_action,
            observation=observation,
            embed_obs=embed_obs,
            action=action,
            action_log_prob=action_log_prob,
            value=value,
            predicted_next_embed_obs=pred_next_embed_obs,
        )

    def sleep(self, observation: Tensor) -> Tensor:
        """Return sleep action."""
        return self.sleep_action.clone()

    def _to_device_dtype(self, tensor_or_module: _tensor_or_module_t) -> _tensor_or_module_t:
        """Send models and tensors to specified device, and cast dtype."""
        return tensor_or_module.to(self.device, self.dtype)

    def _setup_models_and_buffers(self):
        """Setup models and buffer data for interaction."""
        self._to_device_dtype(self.embedding)
        self._to_device_dtype(self.dynamics)
        self._to_device_dtype(self.policy)
        self._to_device_dtype(self.reward)
        self.sleep_action = self._to_device_dtype(self.sleep_action)

        self.embedding.eval()
        self.dynamics.eval()
        self.policy.eval()
        self.reward.eval()

    @torch.no_grad()
    def _embed_observation(self, observation: Tensor) -> Tensor:
        """Embed observation tensor.

        Args:
            observation (Tensor): Observation tensor.

        Returns:
            embed_obs (Tensor): Embedded observation tensor.
        """
        return self.embedding.forward(observation)

    @torch.no_grad()
    def _predict_next_embed_observation(
        self,
        previous_action: Tensor,
        embed_obs: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Predict next embedded observation tensor.

        Args:
            previous_action (Tensor): Previous action tensor.
            embed_obs (Tensor): Embedded observation tensor.
            action (Tensor): Action tensor.

        Returns:
            pred_next_embed_obs (Tensor): Predicted next embedded observation tensor.
        """
        return self.dynamics.forward(previous_action, embed_obs, action)

    @torch.no_grad()
    def _take_actions_and_value(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Through observations to policy.

        Args:
            observation (Tensor): Observation tensor.

        Returns:
            action (Tensor): Action tensor.
            action_log_prob (Tensor): Action log probability tensor.
            value (Tensor): Value tensor.
        """

        action_dist, value = self.policy.forward(observation)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob, value

    @torch.no_grad()
    def _compute_reward(self, predicted_next_embed_obs: Tensor, next_embed_obs: Tensor) -> Tensor:
        """Compute reward.

        Args:
            predicted_next_embed_obs (Tensor): Predicted next embedded observation tensor.
            next_embed_obs (Tensor): Next embedded observation tensor.

        Returns:
            reward (Tensor): Reward tensor.
        """
        return self.reward(predicted_next_embed_obs, next_embed_obs)

    def _clear_step_record(self):
        """Re-create step record object."""
        self.step_record = StepRecord()

    def _store_current_step_data(
        self,
        previous_action: Tensor,
        observation: Tensor,
        embed_obs: Tensor,
        action: Tensor,
        action_log_prob: Tensor,
        value: Tensor,
        predicted_next_embed_obs: Tensor,
    ):
        """Store current step (t) data into step record."""

        self.step_record[RK.PREVIOUS_ACTION] = previous_action  # a_{t-1}
        self.step_record[RK.OBSERVATION] = observation  # o_t
        self.step_record[RK.EMBED_OBSERVATION] = embed_obs  # z_t
        self.step_record[RK.ACTION] = action  # a_t
        self.step_record[RK.ACTION_LOG_PROBABILITY] = action_log_prob  # log \pi(a_t|o_t)
        self.step_record[RK.VALUE] = value  # V(o_t)
        self.step_record[RK.PREDICTED_NEXT_EMBED_OBSERVATION] = predicted_next_embed_obs  # \hat{z}_{t+1}

    def _store_next_step_data(
        self,
        next_observation: Tensor,
        next_embed_obs: Tensor,
        reward: Tensor,
    ):
        """Store next step (t+1) data into step record."""
        self.step_record[RK.NEXT_OBSERVATION] = next_observation  # o_{t+1}
        self.step_record[RK.NEXT_EMBED_OBSERVATION] = next_embed_obs  # z_{t+1}
        self.step_record[RK.REWARD] = reward  # r_{t+1}
