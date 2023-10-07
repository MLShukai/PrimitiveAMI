import pytest
import torch

from src.agents.random_agent import RandomAgent


class TestRandomAgent:
    @pytest.fixture
    def random_agent(self):
        return RandomAgent(action_size=3, mean=0.0, std=1.0, dtype=torch.float32)

    def test_init(self, random_agent):
        assert random_agent.action_size == 3
        assert random_agent.mean == 0.0
        assert random_agent.std == 1.0
        assert random_agent.dtype == torch.float32

    def test_step(self, random_agent):
        out = random_agent.step(None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3,)
        assert out.dtype == torch.float32
