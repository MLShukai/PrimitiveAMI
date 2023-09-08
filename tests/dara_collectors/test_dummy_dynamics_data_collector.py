import pytest
import torch

from src.data_collectors.dummy_dynamics_data_collector import DummyDynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


@pytest.mark.parametrize(
    """
    action_shape,
    observation_shape,
    get_size,
    """,
    [
        ((4, 1), (623,), 3),
        ((32,), (5, 64), 4),
        ((17, 4, 7), (1, 3, 234), 5),
    ],
)
def test_dummy_dynamics_data_collector(action_shape, observation_shape, get_size):
    dummy_dynamics_data_collector = DummyDynamicsDataCollector(action_shape, observation_shape, get_size)

    prev_actions, observations, actions, next_observations = dummy_dynamics_data_collector.get_data()[0:get_size]
    assert prev_actions.size() == (get_size, *action_shape)
    assert observations.size() == (get_size, *observation_shape)
    assert actions.size() == (get_size, *action_shape)
    assert next_observations.size() == (get_size, *observation_shape)
