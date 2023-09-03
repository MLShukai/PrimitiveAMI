import pytest
from pytest_mock import MockerFixture

from src.interactions.interaction import Agent, Environment, Interaction


class InteractionImpl(Interaction):
    def interact(self):
        pass


class TestInteraction:
    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture) -> Agent:
        return mocker.Mock(spec=Agent)

    @pytest.fixture
    def mock_environment(self, mocker: MockerFixture) -> Environment:
        return mocker.Mock(spec=Environment)

    def test_init(self, mock_agent: Agent, mock_environment: Environment):
        interaction = InteractionImpl(mock_agent, mock_environment)
        assert interaction.agent is mock_agent
        assert interaction.environment is mock_environment

    def test_is_abstract(self):
        assert Interaction.__abstractmethods__ == frozenset({"interact"})
