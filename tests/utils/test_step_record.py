from enum import Enum

import pytest

from src.utils.step_record import RecordKeys, StepRecord


class TestStepRecord:
    @pytest.fixture
    def step_record(self) -> StepRecord:
        return StepRecord()

    def test_copy(self, step_record: StepRecord):
        other = step_record.copy()
        assert other == step_record
        assert other is not step_record


class TestRecordKeys:
    def test_enum(self):
        assert issubclass(RecordKeys, (str, Enum))

    def test_enum_values(self):
        assert RecordKeys.PREVIOUS_ACTION == "previous_action"
        assert RecordKeys.OBSERVATION == "observation"
        assert RecordKeys.EMBED_OBSERVATION == "embed_observation"
        assert RecordKeys.ACTION == "action"
        assert RecordKeys.ACTION_LOG_PROBABILITY == "action_log_probability"
        assert RecordKeys.VALUE == "value"
        assert RecordKeys.PREDICTED_NEXT_EMBED_OBSERVATION == "predicted_next_embed_observation"
        assert RecordKeys.NEXT_OBSERVATION == "next_observation"
        assert RecordKeys.NEXT_EMBED_OBSERVATION == "next_embed_observation"
        assert RecordKeys.REWARD == "reward"
