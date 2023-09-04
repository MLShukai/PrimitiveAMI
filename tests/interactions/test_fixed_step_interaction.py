import pytest

from src.agents.agent import Agent
from src.environment.environment import Environment
from src.interactions.fixed_step_interaction import FixedStepInteraction


class TestFixedStepInteraction:
    @pytest.fixture
    def mock_agent(self, mocker):
        mock = mocker.Mock(spec=Agent)
        mock.wakeup.return_value = "initial_action"
        mock.step.return_value = "step_action"
        mock.sleep.return_value = "final_action"
        return mock

    @pytest.fixture
    def mock_environment(self, mocker):
        mock = mocker.Mock(spec=Environment)
        mock.observe.return_value = "some_observation"
        return mock

    @pytest.fixture(scope="function")
    def interaction(self, mock_agent, mock_environment):
        return FixedStepInteraction(mock_agent, mock_environment, 5)

    # Initialization Tests
    def test_initialize_with_action(self, mock_agent, mock_environment, interaction):
        interaction.initialize()

        mock_environment.setup.assert_called_once()
        mock_agent.wakeup.assert_called_once_with("some_observation")
        mock_environment.affect.assert_called_once_with("initial_action")

    def test_initialize_with_none(self, mock_agent, mock_environment, interaction):
        mock_agent.wakeup.return_value = None

        interaction.initialize()

        mock_environment.setup.assert_called_once()
        mock_agent.wakeup.assert_called_once_with("some_observation")
        mock_environment.affect.assert_not_called()

    # Finalization Tests
    def test_finalize_with_action(self, mock_agent, mock_environment, interaction):

        interaction.finalize()

        mock_agent.sleep.assert_called_once_with("some_observation")
        mock_environment.affect.assert_called_once_with("final_action")
        mock_environment.teardown.assert_called_once()

    def test_finalize_with_none(self, mock_agent, mock_environment, interaction):
        mock_agent.sleep.return_value = None

        interaction.finalize()

        mock_agent.sleep.assert_called_once_with("some_observation")
        mock_environment.affect.assert_not_called()
        mock_environment.teardown.assert_called_once()

    # Main Loop Test
    def test_mainloop(self, mock_agent, mock_environment, interaction):

        interaction.mainloop()

        mock_agent.step.assert_called_with("some_observation")
        mock_environment.affect.assert_called_with("step_action")

        assert mock_environment.observe.call_count == 5
        assert mock_agent.step.call_count == 5
        assert mock_environment.affect.call_count == 5

    # Full Interaction Test
    def test_interact(self, mocker, interaction):
        mocker.patch.object(interaction, "initialize")
        mocker.patch.object(interaction, "mainloop")
        mocker.patch.object(interaction, "finalize")

        interaction.interact()

        interaction.initialize.assert_called_once()
        interaction.mainloop.assert_called_once()
        interaction.finalize.assert_called_once()
