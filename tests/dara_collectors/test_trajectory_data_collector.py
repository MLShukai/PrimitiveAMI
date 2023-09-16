import pytest
import torch

from src.data_collectors.trajectory_data_collector import TrajectoryDataCollector
from src.utils.step_record import RecordKeys as RK


@pytest.mark.parametrize(
    """
    max_size,
    gamma,
    gae_lambda,
    observation_shape,
    action_shape,
    num_collect,
    """,
    [(128, 0.99, 0.99, (64, 64), (64, 64), 256)],
)
def test_trajectory_data_collector(max_size, gamma, gae_lambda, observation_shape, action_shape, num_collect):
    trajectory_data_collector = TrajectoryDataCollector(max_size, gamma, gae_lambda)
    for _ in range(num_collect):
        step_record = {}
        observation = torch.randn(*observation_shape)
        step_record[RK.OBSERVATION] = observation
        action = torch.randn(*action_shape)
        step_record[RK.ACTION] = action
        logprob = torch.randn(1)
        step_record[RK.ACTION_LOG_PROBABILITY] = logprob
        reward = torch.randn(1)
        step_record[RK.REWARD] = reward
        value = torch.randn(1)
        step_record[RK.VALUE] = value
        value = torch.randn(1)
        step_record[RK.NEXT_VALUE] = value

        trajectory_data_collector.collect(step_record)

    dataset = trajectory_data_collector.get_data()

    length = min(max_size, num_collect)
    observations, actions, logprobs, advantages, returns, values = dataset[0:length]
    assert observations.size() == (length, *observation_shape)
    assert actions.size() == (length, *action_shape)
    assert logprobs.size() == (length, 1)
    assert advantages.size() == (length, 1)
    assert returns.size() == (length, 1)
    assert values.size() == (length, 1)
