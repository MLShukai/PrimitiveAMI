import pytest
import torch

from src.data_collectors.obs_data_collector import ObsDataCollector as cls
from src.utils.step_record import RecordKeys as RK


class TestVAEDataCollector:
    max_size = 3

    def test__init__(self):
        mod = cls(self.max_size)
        assert mod.max_size is self.max_size
        assert mod.observations == []

    @pytest.fixture
    def dummy_record(self):
        dummy_record = {RK.OBSERVATION: torch.randn(256, 156)}
        return dummy_record

    def test_collect(self, dummy_record):
        # data格納の確認
        mod = cls(self.max_size)
        mod.collect(dummy_record)
        assert len(mod.observations) == 1
        assert torch.equal(mod.observations[0], dummy_record[RK.OBSERVATION])

        # data入れ替えの確認
        for _ in range(5):
            mod.collect(dummy_record)
        assert len(mod.observations) == self.max_size

    def test_get_data(self, dummy_record):
        mod = cls(self.max_size)
        mod.collect(dummy_record)
        assert isinstance(mod.get_data(), torch.utils.data.TensorDataset)
