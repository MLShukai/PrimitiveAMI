from __future__ import annotations

import copy
from collections import UserDict
from enum import Enum


class StepRecord(UserDict):
    """Dictionary that holds the data obtained from one step of the agent."""

    def copy(self) -> StepRecord:
        """Return deep copied Self.

        Returns:
            self (Self): deep copied data.
        """
        return copy.deepcopy(self)


class RecordKeys(str, Enum):
    PREVIOUS_ACTION = "previous_action"  # a_{t-1}
    OBSERVATION = "observation"  # o_t
    EMBED_OBSERVATION = "embed_observation"  # z_t
    ACTION = "action"  # a_t
    ACTION_LOG_PROBABILITY = "action_log_probability"
    VALUE = "value"  # v_t
    PREDICTED_NEXT_EMBED_OBSERVATION = "predicted_next_embed_observation"  # \hat{z}_{t+1}
    NEXT_OBSERVATION = "next_observation"  # o_{t+1}
    NEXT_EMBED_OBSERVATION = "next_embed_observation"  # z_{t+1}
    REWARD = "reward"  # r_{t+1}
