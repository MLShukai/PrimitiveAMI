import pytest

from src.agents.agent import Agent
from src.environment.environment import Environment
from src.interactions.fixed_step_interaction import FixedStepInteraction


class TestFixedStepInteraction:
    @pytest.fixture
    def mock_agent(self, mocker):
        return mocker.Mock(spec=Agent)

    @pytest.fixture
    def mock_environment(self, mocker):
        return mocker.Mock(spec=Environment)

    # Initialization Tests
    def test_initialize_with_action(self, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mock_environment.observe.return_value = "some_observation"
        mock_agent.wakeup.return_value = "initial_action"

        interaction.initialize()

        mock_environment.setup.assert_called_once()
        mock_agent.wakeup.assert_called_once_with("some_observation")
        mock_environment.affect.assert_called_once_with("initial_action")

    def test_initialize_with_none(self, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mock_environment.observe.return_value = "some_observation"
        mock_agent.wakeup.return_value = None

        interaction.initialize()

        mock_environment.setup.assert_called_once()
        mock_agent.wakeup.assert_called_once_with("some_observation")
        mock_environment.affect.assert_not_called()

    # Finalization Tests
    def test_finalize_with_action(self, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mock_environment.observe.return_value = "final_observation"
        mock_agent.sleep.return_value = "final_action"

        interaction.finalize()

        mock_agent.sleep.assert_called_once_with("final_observation")
        mock_environment.affect.assert_called_once_with("final_action")
        mock_environment.teardown.assert_called_once()

    def test_finalize_with_none(self, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mock_environment.observe.return_value = "final_observation"
        mock_agent.sleep.return_value = None

        interaction.finalize()

        mock_agent.sleep.assert_called_once_with("final_observation")
        mock_environment.affect.assert_not_called()
        mock_environment.teardown.assert_called_once()

    # Main Loop Test
    def test_mainloop(self, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mock_environment.observe.return_value = "loop_observation"
        mock_agent.step.return_value = "loop_action"

        interaction.mainloop()

        assert mock_environment.observe.call_count == 5
        assert mock_agent.step.call_count == 5
        assert mock_environment.affect.call_count == 5

    # Full Interaction Test
    def test_interact(self, mocker, mock_agent, mock_environment):
        interaction = FixedStepInteraction(mock_agent, mock_environment, 5)
        mocker.patch.object(interaction, "initialize")
        mocker.patch.object(interaction, "mainloop")
        mocker.patch.object(interaction, "finalize")

        interaction.interact()

        interaction.initialize.assert_called_once()
        interaction.mainloop.assert_called_once()
        interaction.finalize.assert_called_once()
