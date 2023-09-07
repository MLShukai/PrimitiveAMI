import torch

from src.data_collectors.dummy_dynamics_data_collector import DummyDynamicsDataCollector
from src.utils.step_record import RecordKeys as RK


def test_dummy_dynamics_data_collector():
    dummy_dynamics_data_collector = DummyDynamicsDataCollector(64, 128)

    dummy_dynamics_data_collector.collect({})
    dummy_dynamics_data_collector.collect({})
    dummy_dynamics_data_collector.collect({})
    prev_actions, observations, actions, next_observations = dummy_dynamics_data_collector.get_data()
    assert prev_actions.size() == (3, 64)
    assert observations.size() == (3, 128)
    assert actions.size() == (3, 64)
    assert next_observations.size() == (3, 128)

    dummy_dynamics_data_collector.collect({})
    dummy_dynamics_data_collector.collect({})
    prev_actions, observations, actions, next_observations = dummy_dynamics_data_collector.get_data()
    assert prev_actions.size() == (2, 64)
    assert observations.size() == (2, 128)
    assert actions.size() == (2, 64)
    assert next_observations.size() == (2, 128)
