import hydra
from omegaconf import OmegaConf

from src.utils.paths import PROJECT_ROOT


def test_curiosity_ppo_agent(tmp_path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/agent/curiosity_ppo_agent.yaml")
    cfg.logger.save_dir = str(tmp_path)
    hydra.utils.instantiate(cfg)


def test_random_agent():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/agent/random_agent.yaml")
    hydra.utils.instantiate(cfg)
