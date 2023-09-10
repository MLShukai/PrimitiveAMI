import pytest
import torch

from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


@pytest.mark.parametrize(
    """
    max_size,
    observation_shape,
    action_shape,
    """,
    [
        (302, (68,), (73, 4)),
        (400, (62, 5), (33,)),
        (320, (64, 45), (43, 5)),
        (176, (134,), (51,)),
        (32, (68,), (73, 4)),
        (40, (62, 5), (33,)),
        (30, (64, 45), (43, 5)),
        (16, (134,), (51,)),
    ],
)
def test_dynamics_data_collector(max_size, observation_shape, action_shape):
    dynamics_data_collector = DynamicsDataCollector(max_size)
    data = {
        RK.PREVIOUS_ACTION: torch.randn(*action_shape),
        RK.OBSERVATION: torch.randn(*observation_shape),
        RK.ACTION: torch.randn(*action_shape),
        RK.NEXT_OBSERVATION: torch.randn(*observation_shape),
    }

    for _ in range(128):
        dynamics_data_collector.collect(data)

    prev_actions, observations, actions, next_observations = dynamics_data_collector.get_data()[0:max_size]
    assert prev_actions.size(0) <= max_size
    assert prev_actions.size()[1:] == (*action_shape,)
    assert observations.size(0) <= max_size
    assert observations.size()[1:] == (*observation_shape,)
    assert actions.size(0) <= max_size
    assert actions.size()[1:] == (*action_shape,)
    assert next_observations.size(0) <= max_size
    assert next_observations.size()[1:] == (*observation_shape,)
