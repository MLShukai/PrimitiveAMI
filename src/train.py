import logging
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

from src.agents.agent import Agent
from src.data_collectors.aggregations.data_collectors_aggregation import (
    DataCollectorsAggregation,
)
from src.environment.environment import Environment
from src.interactions.interaction import Interaction
from src.models.aggregations.neural_networks import NeuralNetworks
from src.trainers.builders.trainers_builder import TrainersBuilder
from src.trainers.trainer import Trainer
from src.utils.random import seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = logging.getLogger(__name__)


@hydra.main("../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("training start!")

    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)
        logger.info(f"Set seed to {cfg.seed}")

    logger.info(f"Instantiating model: <{cfg.model._target_}>")
    model: NeuralNetworks = hydra.utils.instantiate(cfg.model)

    logger.info(f"Instantiating data collector: <{cfg.data_collector._target_}>")
    data_collectors: DataCollectorsAggregation = hydra.utils.instantiate(cfg.data_collector)

    logger.info(f"Instantiating trainers builder: <{cfg.trainers_builder._target_}>")
    trainers_builder: TrainersBuilder = hydra.utils.instantiate(cfg.trainers_builder)
    trainer = trainers_builder.build(model, data_collectors)

    logger.info(f"Instantiating agent: <{cfg.agent._target_}>")
    agent: Agent = hydra.utils.instantiate(cfg.agent)
    agent = agent(**model.build_agent_models(), data_collector=data_collectors)

    logger.info(f"Instantiating environment: <{cfg.environment._target_}>")
    environment: Environment = hydra.utils.instantiate(cfg.environment)

    logger.info(f"Instantiating interaction<{cfg.interaction._target_}>")
    interaction: Interaction = hydra.utils.instantiate(cfg.interaction)
    interaction = interaction(agent=agent, environment=environment)

    loop(interaction, trainer)

    environment.teardown()

    if cfg.save_data_collectors_state:
        data_collectors.save_state_dict_to_files(Path(cfg.paths.data_collectors_dir))
        logger.info("Saved data collector states.")

    logger.info("End training.")


def loop(interaction: Interaction, trainer: Trainer) -> None:
    """main loop process."""
    logger.info("Start main loop.")

    try:
        while True:
            logger.info("Interacting...")
            interaction.interact()
            logger.info("End iteraction.")

            logger.info("Training...")
            trainer.train()
            logger.info("End training.")
    except KeyboardInterrupt:
        logger.error("Keyboard interrupted.")
    except Exception as e:
        logger.exception(e)
    finally:
        logger.info("End main loop.")


if __name__ == "__main__":
    main()
