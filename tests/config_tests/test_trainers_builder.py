from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_inverse_forward_ppo(tmp_path: Path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/trainers_builder/inverse_forward_ppo.yaml")

    with open_dict(cfg):
        cfg.inverse_dynamics.pl_trainer.default_root_dir = str(tmp_path / "inverse_dynamics")
        cfg.forward_dynamics.pl_trainer.default_root_dir = str(tmp_path / "forward_dynamics")
        cfg.ppo.pl_trainer.default_root_dir = str(tmp_path / "ppo")

    hydra.utils.instantiate(cfg)
