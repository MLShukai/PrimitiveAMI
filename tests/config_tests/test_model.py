import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_inverse_forward_ppo():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/inverse_forward_ppo.yaml")

    with open_dict(cfg):
        height, width = 256, 256
        cfg.inverse_dynamics.inverse_dynamics_net.observation_encoder.height = height
        cfg.inverse_dynamics.inverse_dynamics_net.observation_encoder.width = width
        cfg.ppo.net.base_model.modules[0].height = height
        cfg.ppo.net.base_model.modules[0].width = width

    hydra.utils.instantiate(cfg)


def test_vae_forward_ppo():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/vae_forward_ppo.yaml")
    with open_dict(cfg):
        height, width = 256, 256
        cfg.vae.vae_net.encoder.base_model.height = height
        cfg.vae.vae_net.encoder.base_model.width = width
        cfg.ppo.net.base_model.modules[0].height = height
        cfg.ppo.net.base_model.modules[0].width = width

    hydra.utils.instantiate(cfg)


def test_vae_time_series_forward_ppo():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/vae_time_series_forward_ppo.yaml")
    with open_dict(cfg):
        height, width = 256, 256
        cfg.vae.vae_net.encoder.base_model.height = height
        cfg.vae.vae_net.encoder.base_model.width = width
        cfg.ppo.net.base_model.modules[0].height = height
        cfg.ppo.net.base_model.modules[0].width = width

    hydra.utils.instantiate(cfg)
