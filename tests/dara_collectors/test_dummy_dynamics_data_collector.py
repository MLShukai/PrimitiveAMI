import pytest
import torch

from src.data_collectors.dummy_dynamics_data_collector import DummyDynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


@pytest.mark.parametrize(
    """
    action_dim,
    observation_dim,
    get_size,
    """,
    [
        (4, 623, 3),
        (32, 64, 4),
        (17, 234, 5),
    ],
)
def test_dummy_dynamics_data_collector(action_dim, observation_dim, get_size):
    dummy_dynamics_data_collector = DummyDynamicsDataCollector(action_dim, observation_dim, get_size)

    prev_actions, observations, actions, next_observations = dummy_dynamics_data_collector.get_data()
    assert prev_actions.size() == (get_size, action_dim)
    assert observations.size() == (get_size, observation_dim)
    assert actions.size() == (get_size, action_dim)
    assert next_observations.size() == (get_size, observation_dim)
