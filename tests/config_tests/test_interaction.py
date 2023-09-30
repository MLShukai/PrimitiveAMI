import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_fixed_step():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/interaction/fixed_step.yaml")
    hydra.utils.instantiate(cfg)
