import pytest
import torch

from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


@pytest.mark.parametrize(
    """
    decay_prob,
    max_size,
    observation_dim,
    action_dim,
    """,
    [
        (0, 302, 68, 73),
        (0.01, 400, 62, 33),
        (0.99, 320, 64, 43),
        (1, 176, 134, 51),
    ],
)
def test_dynamics_data_collector(decay_prob, max_size, observation_dim, action_dim):
    dynamics_data_collector = DynamicsDataCollector(decay_prob, max_size)
    data = {
        RK.PREVIOUS_ACTION: torch.randn(action_dim),
        RK.OBSERVATION: torch.randn(observation_dim),
        RK.ACTION: torch.randn(action_dim),
        RK.NEXT_OBSERVATION: torch.randn(observation_dim),
    }

    for _ in range(128):
        dynamics_data_collector.collect(data)

    prev_actions, observations, actions, next_observations = dynamics_data_collector.get_data()
    assert prev_actions.size()[0] <= max_size
    assert prev_actions.size()[1] == action_dim
    assert observations.size()[0] <= max_size
    assert observations.size()[1] == observation_dim
    assert actions.size()[0] <= max_size
    assert actions.size()[1] == action_dim
    assert next_observations.size()[0] <= max_size
    assert next_observations.size()[1] == observation_dim
