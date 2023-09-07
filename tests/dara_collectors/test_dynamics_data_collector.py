import torch

from src.data_collectors.dynamics_data_collector import DynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


def test_dynamics_data_collector():
    dynamics_data_collector = DynamicsDataCollector()
    data = {
        RK.PREVIOUS_ACTION: torch.randn(64),
        RK.OBSERVATION: torch.randn(128),
        RK.ACTION: torch.randn(64),
        RK.NEXT_OBSERVATION: torch.randn(128),
    }

    dynamics_data_collector.collect(data)
    dynamics_data_collector.collect(data)
    dynamics_data_collector.collect(data)
    prev_actions, observations, actions, next_observations = dynamics_data_collector.get_data()
    assert prev_actions.size() == (3, 64)
    assert observations.size() == (3, 128)
    assert actions.size() == (3, 64)
    assert next_observations.size() == (3, 128)

    dynamics_data_collector.collect(data)
    dynamics_data_collector.collect(data)
    prev_actions, observations, actions, next_observations = dynamics_data_collector.get_data()
    assert prev_actions.size() == (2, 64)
    assert observations.size() == (2, 128)
    assert actions.size() == (2, 64)
    assert next_observations.size() == (2, 128)
