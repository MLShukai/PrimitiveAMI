"""This demo shows the activities of un-learned agent."""

import logging
import pprint
import time
from argparse import ArgumentParser, Namespace

import colorlog
import torch
import torch.nn as nn

from src.agents.curiosity_ppo_agent import CuriosityPPOAgent
from src.data_collectors.empty_data_collector import EmptyDataCollector
from src.environment.interval_adjustors.sleep_interval_adjustor import (
    SleepIntervalAdjustor,
)
from src.environment.periodic_environment import PeriodicEnvironment
from src.interactions.fixed_step_interaction import FixedStepInteraction
from src.models.components.forward_dynamics.dense_net_forward_dynamics import (
    DenseNetForwardDynamics,
)
from src.models.components.fully_connected import FullyConnected
from src.models.components.policy.tanh_normal_stochastic_policy import (
    TanhNormalStochasticPolicy,
)
from src.models.components.policy_value_common_net import PolicyValueCommonNet
from src.models.components.reward.curiosity_reward import CuriosityReward
from src.models.components.small_conv_net import SmallConvNet
from src.models.components.value.fully_connect_value import FullyConnectValue
from src.utils.environment import create_frame_sensor, create_locomotion_actuator
from src.utils.random import seed_everything

logger = logging.getLogger(__name__)


def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_root_logger(args)
    logger.info("Configured Logger.")
    logger.info(f"\nArgs: {pprint.pformat(args.__dict__)}")

    seed_everything(args.random_seed)
    logger.info(f"Set random seed: {args.random_seed}")

    environment = create_environment(args)
    logger.info("Created Environment.")

    agent = create_agent(args)
    logger.info("Create Agent.")

    interaction = FixedStepInteraction(agent, environment, args.num_steps)
    logger.info("Create Interaction")

    try:
        logger.info(f"Interact {args.num_steps} steps.")
        start_time = time.perf_counter()
        interaction.interact()
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Interaction end, {args.num_steps/elapsed_time:.2f} fps.")
    except KeyboardInterrupt:
        logger.error("Interrupted.")
    except Exception as e:
        logger.exception(e)
    finally:
        environment.actuator.operate(agent._postprocess_action(agent.sleep_action))

    logger.info("End agent interaction demo.")


def get_parser() -> ArgumentParser:
    """Define command line arguments for configuration of this demo."""
    parser = ArgumentParser()

    # Logging
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO"
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="%(log_color)s%(asctime)s.%(msecs)03d %(name)s [%(levelname)s]: %(reset)s %(blue)s%(message)s",
    )
    parser.add_argument("--log-datefmt", type=str, default="%Y/%m/%d %H:%M:%S")

    # Environment
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--osc-address", type=str, default="127.0.0.1")
    parser.add_argument("--osc-sender-port", type=int, default=9000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--base-cam-fps", type=float, default=60.0)
    parser.add_argument("--frame-width", type=int, default=256)
    parser.add_argument("--frame-height", type=int, default=256)
    parser.add_argument("--frame-channels", type=int, default=3)

    # Agent
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--action-size", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="float32")

    # Interaction
    parser.add_argument("--num-steps", type=int, default=60)

    # Reproducing
    parser.add_argument("--random-seed", type=int, default=9)

    return parser


def setup_root_logger(args: Namespace) -> None:
    """Setup root logger system for logging."""
    root_logger = colorlog.getLogger()
    sh = colorlog.StreamHandler()
    fmtter = colorlog.ColoredFormatter(
        args.log_format,
        args.log_datefmt,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    sh.setFormatter(fmtter)
    root_logger.addHandler(sh)
    root_logger.setLevel(args.log_level)


def create_environment(args: Namespace) -> PeriodicEnvironment:

    # Sensor
    sensor = create_frame_sensor(
        args.camera_index,
        args.frame_width,
        args.frame_height,
        args.base_cam_fps,
        True,
    )
    # Actuator
    actuator = create_locomotion_actuator(args.osc_address, args.osc_sender_port)

    # IntervalAdjustor
    adjustor = SleepIntervalAdjustor(1 / args.fps, offset=0.0)

    # Environment
    environment = PeriodicEnvironment(sensor, actuator, adjustor)
    return environment


def create_agent(args: Namespace):  # -> CuriosityPPOAgent:

    height, width, channels = args.frame_width, args.frame_height, args.frame_channels
    embed_dim = args.embed_dim
    action_size = args.action_size
    device = torch.device(args.device)
    dtype = getattr(torch, args.precision)
    # Observation Encoder
    obs_encoder = SmallConvNet(height, width, channels, embed_dim)
    # ForwardDynamics
    dynamics = DenseNetForwardDynamics(action_size, embed_dim)
    # PolicyValueCommonNet
    base_model = nn.Sequential(
        SmallConvNet(height, width, channels, embed_dim),
        FullyConnected(embed_dim, embed_dim),
        FullyConnected(embed_dim, embed_dim),
    )
    policy = TanhNormalStochasticPolicy(embed_dim, action_size)
    value = FullyConnectValue(embed_dim)
    policy_value = PolicyValueCommonNet(base_model, policy, value)
    # RewardModel
    reward = CuriosityReward()
    # Data Collector
    data_collector = EmptyDataCollector()
    # Sleep Action
    sleep_action = torch.zeros(action_size)

    # Agemt
    agent = CuriosityPPOAgent(
        obs_encoder,
        dynamics,
        policy_value,
        reward,
        data_collector,
        sleep_action,
        device,
        dtype,
    )
    return agent


if __name__ == "__main__":
    main()
