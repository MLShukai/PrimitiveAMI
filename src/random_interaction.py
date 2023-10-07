import logging

import hydra
import rootutils
from omegaconf import DictConfig

from src.agents.agent import Agent
from src.environment.environment import Environment
from src.interactions.interaction import Interaction
from src.utils.random import seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = logging.getLogger(__name__)


@hydra.main("../configs", config_name="random_interaction.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Random interaction start!")

    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)
        logger.info(f"Set seed to {cfg.seed}")

    logger.info(f"Instantiating agent: <{cfg.agent._target_}>")
    agent: Agent = hydra.utils.instantiate(cfg.agent)

    logger.info(f"Instantiating environment: <{cfg.environment._target_}>")
    environment: Environment = hydra.utils.instantiate(cfg.environment)

    logger.info(f"Instantiating interaction<{cfg.interaction._target_}>")
    interaction: Interaction = hydra.utils.instantiate(cfg.interaction)
    interaction = interaction(agent=agent, environment=environment)

    loop(interaction)

    environment.teardown()

    logger.info("End random interaction.")


def loop(interaction: Interaction) -> None:
    """main loop process."""
    logger.info("Start main loop.")

    try:
        num_interact = 0
        while True:
            logger.info("Interacting...")
            interaction.interact()
            num_interact += 1
            logger.info(f"End iteraction {num_interact} times.")

    except KeyboardInterrupt:
        logger.error("Keyboard interrupted.")
    except Exception as e:
        logger.exception(e)
    finally:
        logger.info("End main loop.")


if __name__ == "__main__":
    main()
