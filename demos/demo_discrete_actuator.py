"""Demo script for testing DiscreteActuator class."""
import time

import torch
from pythonosc.udp_client import SimpleUDPClient
from vrchat_io.controller.osc import InputController
from vrchat_io.controller.wrappers.osc import MultiInputWrapper

from src.environment.actuators.discrete_actuator import DiscreteActuator

if __name__ == "__main__":

    actions = torch.tensor(
        [
            [1, 2, 1, 0, 0],  # forward, left, look right
            [2, 1, 2, 0, 1],  # Run backward, right, look left
            [0, 0, 0, 0, 0],  # stop all.
            [1, 0, 0, 1, 1],  # Jump and run forward
        ],
        dtype=torch.long,
    )

    actuator = DiscreteActuator(MultiInputWrapper(InputController(SimpleUDPClient("127.0.0.1", 9000))))

    for action in actions:
        print(f"Do {action.tolist()}")
        actuator.operate(action)
        time.sleep(1)

    actuator.teardown()
