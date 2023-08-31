import pytest

from src.utils.step_record import StepRecord


class TestStepRecord:
    @pytest.fixture
    def step_record(self) -> StepRecord:
        return StepRecord()

    def test_copy(self, step_record: StepRecord):
        other = step_record.copy()
        assert other == step_record
        assert other is not step_record
