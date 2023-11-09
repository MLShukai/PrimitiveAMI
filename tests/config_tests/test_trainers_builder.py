from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_inverse_forward_ppo(tmp_path: Path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/trainers_builder/inverse_forward_ppo.yaml")

    with open_dict(cfg):
        cfg.inverse_dynamics.pl_trainer.default_root_dir = str(tmp_path / "inverse_dynamics")
        cfg.inverse_dynamics.pl_trainer.accelerator = "cpu"
        cfg.forward_dynamics.pl_trainer.default_root_dir = str(tmp_path / "forward_dynamics")
        cfg.forward_dynamics.pl_trainer.accelerator = "cpu"
        cfg.ppo.pl_trainer.default_root_dir = str(tmp_path / "ppo")
        cfg.ppo.pl_trainer.accelerator = "cpu"

    hydra.utils.instantiate(cfg)


def test_vae_forward_ppo(tmp_path: Path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/trainers_builder/vae_forward_ppo.yaml")

    with open_dict(cfg):
        cfg.vae_trainer.pl_trainer.default_root_dir = str(tmp_path / "vae")
        cfg.vae_trainer.pl_trainer.accelerator = "cpu"
        cfg.forward_dynamics_trainer.pl_trainer.default_root_dir = str(tmp_path / "forward_dynamics")
        cfg.forward_dynamics_trainer.pl_trainer.accelerator = "cpu"
        cfg.ppo_trainer.pl_trainer.default_root_dir = str(tmp_path / "ppo")
        cfg.ppo_trainer.pl_trainer.accelerator = "cpu"

    hydra.utils.instantiate(cfg)


def test_vae_time_series_forward_ppo(tmp_path: Path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/trainers_builder/vae_time_series_forward_ppo.yaml")

    with open_dict(cfg):
        cfg.vae_trainer.pl_trainer.default_root_dir = str(tmp_path / "vae")
        cfg.vae_trainer.pl_trainer.accelerator = "cpu"
        cfg.forward_dynamics_trainer.pl_trainer.default_root_dir = str(tmp_path / "forward_dynamics")
        cfg.forward_dynamics_trainer.pl_trainer.accelerator = "cpu"
        cfg.ppo_trainer.pl_trainer.default_root_dir = str(tmp_path / "ppo")
        cfg.ppo_trainer.pl_trainer.accelerator = "cpu"

    hydra.utils.instantiate(cfg)
