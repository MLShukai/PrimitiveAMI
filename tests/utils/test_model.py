import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.utils.model import MultiCategoricals, SequentialModuleList


def test_SequentialModuleList():
    mod = SequentialModuleList([nn.Linear(2, 2) for _ in range(3)])
    mod(torch.randn(2))


class TestMultiCategoricals:
    @pytest.fixture
    def distributions(self) -> list[Categorical]:
        choices_per_dist = [3, 2, 5]
        batch_size = 8
        return [Categorical(logits=torch.zeros(batch_size, c)) for c in choices_per_dist]

    @pytest.fixture
    def multi_categoricals(self, distributions) -> MultiCategoricals:
        return MultiCategoricals(distributions)

    def test_init(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.batch_shape == (8, 3)

    def test_sample(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.sample().shape == (8, 3)
        assert multi_categoricals.sample((1, 2)).shape == (1, 2, 8, 3)

    def test_log_prob(self, multi_categoricals: MultiCategoricals):
        sampled = multi_categoricals.sample()
        assert multi_categoricals.log_prob(sampled).shape == sampled.shape

    def test_entropy(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.entropy().shape == (8, 3)
