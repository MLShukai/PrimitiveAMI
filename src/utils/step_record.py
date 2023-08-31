from __future__ import annotations

import copy
from collections import UserDict


class StepRecord(UserDict):
    """Dictionary that holds the data obtained from one step of the agent."""

    def copy(self) -> StepRecord:
        """Return deep copied Self.

        Returns:
            self (Self): deep copied data.
        """
        return copy.deepcopy(self)
