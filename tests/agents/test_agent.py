import pytest

from src.agents.agent import Agent


class AgentImpl(Agent):
    def step(self, observation):
        return observation


class TestAgent:
    @pytest.fixture
    def agent(self):
        return AgentImpl()

    def test_instantiate_error(self):
        with pytest.raises(TypeError):
            Agent()

    def test_wakeup(self, agent):
        assert agent.wakeup(0) is None

    def test_step(self, agent):
        assert agent.step(0) == 0

    def test_sleep(self, agent):
        assert agent.sleep(0) is None
