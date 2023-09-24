import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_inverse_forward_ppo():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/inverse_forward_ppo.yaml")

    with open_dict(cfg):
        height, width = 256, 256
        cfg.inverse_dynamics.inverse_dynamics_net.observation_encoder.height = height
        cfg.inverse_dynamics.inverse_dynamics_net.observation_encoder.width = width
        cfg.ppo.net.base_model.height = height
        cfg.ppo.net.base_model.width = width

    hydra.utils.instantiate(cfg)