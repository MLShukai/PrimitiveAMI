import hydra
from omegaconf import OmegaConf

from src.utils.paths import PROJECT_ROOT


def test_curiosity_ppo_agent(tmp_path):
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/agent/curiosity_ppo_agent.yaml")
    cfg.logger.save_dir = str(tmp_path)
    hydra.utils.instantiate(cfg)
