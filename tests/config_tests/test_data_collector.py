import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_dict_dynamics_trajectory():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/data_collector/dict_dynamics_trajectory.yaml")
    with open_dict(cfg):
        cfg.trajectory.max_size = 128

    hydra.utils.instantiate(cfg)


def test_dict_observation_dynamics_trajectory():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/data_collector/dict_observation_dynamics_trajectory.yaml")
    with open_dict(cfg):
        cfg.trajectory.max_size = 128

    hydra.utils.instantiate(cfg)


def test_dict_observation_time_series_dynamics_trajectory():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/data_collector/dict_observation_time_series_dynamics_trajectory.yaml")
    with open_dict(cfg):
        cfg.trajectory.max_size = 128

    hydra.utils.instantiate(cfg)
