import hydra
from omegaconf import OmegaConf, open_dict

from src.utils.paths import PROJECT_ROOT


def test_curiosity_ppo_agent():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/agent/curiosity_ppo_agent.yaml")
    hydra.utils.instantiate(cfg)
