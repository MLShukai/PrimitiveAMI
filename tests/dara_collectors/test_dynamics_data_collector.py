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
def test_dynamics_data_collector(max_size, observation_shape, action_shape, tmp_path):
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

    # test state dict
    state_dict = dynamics_data_collector.state_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["max_size"] == max_size
    assert state_dict["prev_actions"] == dynamics_data_collector.prev_actions
    assert state_dict["observations"] == dynamics_data_collector.observations
    assert state_dict["actions"] == dynamics_data_collector.actions
    assert state_dict["next_observations"] == dynamics_data_collector.next_observations

    # test load state dict
    empty_states = {
        "max_size": 0,
        "prev_actions": [],
        "observations": [],
        "actions": [],
        "next_observations": [],
    }

    dynamics_data_collector.load_state_dict(empty_states)
    assert dynamics_data_collector.prev_actions == []
    assert dynamics_data_collector.observations == []
    assert dynamics_data_collector.actions == []
    assert dynamics_data_collector.next_observations == []

    # test save to file
    dynamics_data_collector.load_state_dict(state_dict)
    dist_path = tmp_path / "dynamics.pkl"
    dynamics_data_collector.save_to_file(dist_path)
    assert dist_path.exists()

    # test load from file
    dynamics_data_collector = DynamicsDataCollector.load_from_file(dist_path, max_size=10)
