import pytest
import torch
from pytest_mock import MockerFixture
from torch.distributions import Normal

from src.agents.curiosity_ppo_agent import (
    RK,
    CuriosityPPOAgent,
    CuriosityReward,
    DataCollector,
    ForwardDynamics,
    ObservationEncoder,
    PolicyValueCommonNet,
    StepRecord,
)

SLEEP_ACTION = torch.zeros(3)


class TestCuriosityPPOAgent:
    @pytest.fixture
    def mock_embedding(self, mocker: MockerFixture) -> ObservationEncoder:
        mock = mocker.Mock(spec=ObservationEncoder)
        mock.forward.return_value = torch.tensor(0.0)
        return mock

    @pytest.fixture
    def mock_forward_dynamics(self, mocker: MockerFixture) -> ForwardDynamics:
        mock = mocker.Mock(spec=ForwardDynamics)
        mock.forward.return_value = torch.tensor(0.0)
        return mock

    @pytest.fixture
    def mock_policy(self, mocker: MockerFixture) -> PolicyValueCommonNet:
        mock = mocker.Mock(spec=PolicyValueCommonNet)
        mock.forward.return_value = (Normal(torch.tensor(0.0), torch.tensor(1.0)), torch.tensor(0.0))
        return mock

    @pytest.fixture
    def mock_reward(self, mocker: MockerFixture) -> CuriosityReward:
        mock = mocker.Mock(spec=CuriosityReward)
        mock.reward.return_value = torch.tensor(0.0)
        return mock

    @pytest.fixture
    def mock_data_collector(self, mocker: MockerFixture) -> DataCollector:
        mock = mocker.Mock(spec=DataCollector)
        return mock

    @pytest.fixture
    def curiosity_ppo_agent(
        self,
        mock_embedding: ObservationEncoder,
        mock_forward_dynamics: ForwardDynamics,
        mock_policy: PolicyValueCommonNet,
        mock_reward: CuriosityReward,
        mock_data_collector: DataCollector,
    ) -> CuriosityPPOAgent:
        return CuriosityPPOAgent(
            mock_embedding, mock_forward_dynamics, mock_policy, mock_reward, mock_data_collector, SLEEP_ACTION
        )

    @pytest.fixture
    def curiosity_ppo_agent_patch_method(self, mocker: MockerFixture, curiosity_ppo_agent: CuriosityPPOAgent):
        mocker.patch.object(curiosity_ppo_agent, "_setup_models_and_buffers")
        mocker.patch.object(curiosity_ppo_agent, "_embed_observation").return_value = torch.tensor(0.0)
        mocker.patch.object(curiosity_ppo_agent, "_take_actions_and_value").return_value = (
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
        )
        mocker.patch.object(curiosity_ppo_agent, "_predict_next_embed_observation").return_value = torch.tensor(4.0)
        mocker.patch.object(curiosity_ppo_agent, "_clear_step_record")
        mocker.patch.object(curiosity_ppo_agent, "_store_current_step_data")
        mocker.patch.object(curiosity_ppo_agent, "_compute_reward").return_value = torch.tensor(6.0)
        mocker.patch.object(curiosity_ppo_agent, "_store_next_step_data")
        return curiosity_ppo_agent

    def test_init(
        self,
        curiosity_ppo_agent: CuriosityPPOAgent,
        mock_embedding: ObservationEncoder,
        mock_forward_dynamics: ForwardDynamics,
        mock_policy: PolicyValueCommonNet,
        mock_reward: CuriosityReward,
        mock_data_collector: DataCollector,
    ):
        assert curiosity_ppo_agent.embedding == mock_embedding
        assert curiosity_ppo_agent.dynamics == mock_forward_dynamics
        assert curiosity_ppo_agent.policy == mock_policy
        assert curiosity_ppo_agent.reward == mock_reward
        assert curiosity_ppo_agent.data_collector == mock_data_collector
        torch.testing.assert_close(curiosity_ppo_agent.sleep_action, SLEEP_ACTION)
        assert curiosity_ppo_agent.device == "cpu"
        assert curiosity_ppo_agent.dtype == torch.float32
        assert curiosity_ppo_agent.step_record == StepRecord()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_to_device_dtype(self, curiosity_ppo_agent: CuriosityPPOAgent):
        curiosity_ppo_agent.device = torch.device("cuda:0")
        curiosity_ppo_agent.dtype = torch.float64

        dummy_tensor = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        dummy_tensor = curiosity_ppo_agent._to_device_dtype(dummy_tensor)

        assert dummy_tensor.device == torch.device("cuda:0")
        assert dummy_tensor.dtype == torch.float64

    def test_setup_models_and_buffers(self, mocker: MockerFixture, curiosity_ppo_agent: CuriosityPPOAgent):
        curiosity_ppo_agent._setup_models_and_buffers()

        device, dtype = curiosity_ppo_agent.device, curiosity_ppo_agent.dtype
        curiosity_ppo_agent.embedding.to.assert_called_with(device, dtype)
        curiosity_ppo_agent.dynamics.to.assert_called_with(device, dtype)
        curiosity_ppo_agent.policy.to.assert_called_with(device, dtype)
        curiosity_ppo_agent.reward.to.assert_called_with(device, dtype)

        curiosity_ppo_agent.embedding.eval.assert_called_with()
        curiosity_ppo_agent.dynamics.eval.assert_called_with()
        curiosity_ppo_agent.policy.eval.assert_called_with()
        curiosity_ppo_agent.reward.eval.assert_called_with()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_setup_models_and_buffers_cuda(self, curiosity_ppo_agent: CuriosityPPOAgent):
        curiosity_ppo_agent.device = torch.device("cuda:0")
        curiosity_ppo_agent.dtype = torch.float64

        curiosity_ppo_agent._setup_models_and_buffers()

        assert curiosity_ppo_agent.sleep_action.device == torch.device("cuda:0")
        assert curiosity_ppo_agent.sleep_action.dtype == torch.float64

    def test_embed_observation(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_observation = torch.tensor(1.0)
        assert curiosity_ppo_agent._embed_observation(dummy_observation) == torch.tensor(0.0)
        curiosity_ppo_agent.embedding.forward.assert_called_with(dummy_observation)

    def test_predict_next_embed_observation(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_prev_action = torch.tensor(1.0)
        dummy_embed_obs = torch.tensor(2.0)
        dummy_action = torch.tensor(3.0)
        assert curiosity_ppo_agent._predict_next_embed_observation(
            dummy_prev_action, dummy_embed_obs, dummy_action
        ) == torch.tensor(0.0)
        curiosity_ppo_agent.dynamics.forward.assert_called_with(dummy_prev_action, dummy_embed_obs, dummy_action)

    def test_compute_reward(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_embed_obs = torch.tensor(1.0)
        dummy_pred_next_embed_obs = torch.tensor(2.0)
        assert curiosity_ppo_agent._compute_reward(dummy_embed_obs, dummy_pred_next_embed_obs) == torch.tensor(0.0)
        curiosity_ppo_agent.reward.reward.assert_called_with(dummy_embed_obs, dummy_pred_next_embed_obs)

    def test_clear_step_record(self, curiosity_ppo_agent: CuriosityPPOAgent):
        curiosity_ppo_agent._clear_step_record()
        assert curiosity_ppo_agent.step_record == StepRecord()

    def test_store_current_step_data(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_previous_action = torch.tensor(1.0)
        dummy_observation = torch.tensor(2.0)
        dummy_embed_obs = torch.tensor(3.0)
        dummy_action = torch.tensor(4.0)
        dummy_action_log_prob = torch.tensor(5.0)
        dummy_value = torch.tensor(6.0)
        dummy_pred_next_embed_obs = torch.tensor(7.0)

        curiosity_ppo_agent._store_current_step_data(
            dummy_previous_action,
            dummy_observation,
            dummy_embed_obs,
            dummy_action,
            dummy_action_log_prob,
            dummy_value,
            dummy_pred_next_embed_obs,
        )

        assert curiosity_ppo_agent.step_record[RK.PREVIOUS_ACTION] == dummy_previous_action
        assert curiosity_ppo_agent.step_record[RK.OBSERVATION] == dummy_observation
        assert curiosity_ppo_agent.step_record[RK.EMBED_OBSERVATION] == dummy_embed_obs
        assert curiosity_ppo_agent.step_record[RK.ACTION] == dummy_action
        assert curiosity_ppo_agent.step_record[RK.ACTION_LOG_PROBABILITY] == dummy_action_log_prob
        assert curiosity_ppo_agent.step_record[RK.VALUE] == dummy_value
        assert curiosity_ppo_agent.step_record[RK.PREDICTED_NEXT_EMBED_OBSERVATION] == dummy_pred_next_embed_obs

    def test_store_next_step_data(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_observation = torch.tensor(1.0)
        dummy_embed_obs = torch.tensor(2.0)
        dummy_reward = torch.tensor(3.0)
        dummy_next_value = torch.tensor(4.0)

        curiosity_ppo_agent._store_next_step_data(dummy_observation, dummy_embed_obs, dummy_reward, dummy_next_value)

        assert curiosity_ppo_agent.step_record[RK.REWARD] == dummy_reward
        assert curiosity_ppo_agent.step_record[RK.NEXT_OBSERVATION] == dummy_observation
        assert curiosity_ppo_agent.step_record[RK.NEXT_EMBED_OBSERVATION] == dummy_embed_obs
        assert curiosity_ppo_agent.step_record[RK.NEXT_VALUE] == dummy_next_value

    def test_collect_data(self, curiosity_ppo_agent: CuriosityPPOAgent):
        dummy_data = torch.zeros(1, 8)
        curiosity_ppo_agent.step_record["dummy"] = dummy_data
        curiosity_ppo_agent._collect_data()

    def test_wakeup(self, curiosity_ppo_agent_patch_method: CuriosityPPOAgent):
        dummy_observation = torch.tensor(5.0)
        assert curiosity_ppo_agent_patch_method.wakeup(dummy_observation) == torch.tensor(1.0)

        curiosity_ppo_agent_patch_method._setup_models_and_buffers.assert_called_once()
        curiosity_ppo_agent_patch_method._embed_observation.assert_called_with(dummy_observation)
        curiosity_ppo_agent_patch_method._take_actions_and_value.assert_called_with(dummy_observation)
        curiosity_ppo_agent_patch_method._predict_next_embed_observation.assert_called_with(
            SLEEP_ACTION, torch.tensor(0.0), torch.tensor(1.0)
        )
        curiosity_ppo_agent_patch_method._clear_step_record.assert_called_once()
        curiosity_ppo_agent_patch_method._store_current_step_data.assert_called_with(
            previous_action=SLEEP_ACTION,
            observation=dummy_observation,
            embed_obs=torch.tensor(0.0),
            action=torch.tensor(1.0),
            action_log_prob=torch.tensor(2.0),
            value=torch.tensor(3.0),
            predicted_next_embed_obs=torch.tensor(4.0),
        )

    def test_step(self, curiosity_ppo_agent_patch_method: CuriosityPPOAgent):
        curiosity_ppo_agent_patch_method.step_record[RK.PREDICTED_NEXT_EMBED_OBSERVATION] = torch.tensor(4.0)
        curiosity_ppo_agent_patch_method.step_record[RK.ACTION] = torch.tensor(-1.0)
        dummy_observation = torch.tensor(5.0)
        assert curiosity_ppo_agent_patch_method.step(dummy_observation) == torch.tensor(1.0)

        curiosity_ppo_agent_patch_method._embed_observation.assert_called_with(dummy_observation)
        curiosity_ppo_agent_patch_method._compute_reward.assert_called_with(torch.tensor(4.0), torch.tensor(0.0))
        curiosity_ppo_agent_patch_method._store_next_step_data.assert_called_with(
            dummy_observation, torch.tensor(0.0), torch.tensor(6.0), torch.tensor(3.0)
        )
        curiosity_ppo_agent_patch_method.data_collector.collect.assert_called_with(
            curiosity_ppo_agent_patch_method.step_record.copy()
        )
        curiosity_ppo_agent_patch_method._take_actions_and_value.assert_called_with(dummy_observation)
        curiosity_ppo_agent_patch_method._predict_next_embed_observation.assert_called_with(
            torch.tensor(-1.0), torch.tensor(0.0), torch.tensor(1.0)
        )
        curiosity_ppo_agent_patch_method._store_current_step_data.assert_called_with(
            previous_action=torch.tensor(-1.0),
            observation=dummy_observation,
            embed_obs=torch.tensor(0.0),
            action=torch.tensor(1.0),
            action_log_prob=torch.tensor(2.0),
            value=torch.tensor(3.0),
            predicted_next_embed_obs=torch.tensor(4.0),
        )

    def test_sleep(self, curiosity_ppo_agent: CuriosityPPOAgent):
        torch.testing.assert_close(curiosity_ppo_agent.sleep(torch.tensor(0.0)), SLEEP_ACTION)
