"""This file contains helper tools for creating environment components."""
from typing import Optional

import cv2
from pythonosc.udp_client import SimpleUDPClient
from vrchat_io.controller.osc import InputController
from vrchat_io.controller.wrappers.osc import (
    AXES_LOCOMOTION_RESET_VALUES,
    AxesLocomotionWrapper,
    MultiInputWrapper,
)
from vrchat_io.vision import OpenCVVideoCapture
from vrchat_io.vision.wrappers import RatioCropWrapper, ResizeWrapper

from src.environment.actuators.locomotion_actuator import LocomotionActuator
from src.environment.sensors.frame_sensor import FrameSensor


def create_frame_sensor(
    camera_index: int = 0,
    width: int = 512,
    height: int = 512,
    base_fps: float = 60.0,
    bgr2rgb: bool = True,
    aspect_ratio: Optional[float] = None,
) -> FrameSensor:
    """Create FrameSensor object.

    Args:
        camera_index (int): Capture device index.
        width (int): Width of captured frames.
        height (int): Height of captured frames.
        base_fps (float): Base framerate of capture device.
        bgr2rgb (bool): Convert BGR to RGB.
        aspect_ratio (Optional[float]): Ratio to crop frames to. If None, use `width/height`.
    """
    cam = OpenCVVideoCapture(
        camera=cv2.VideoCapture(camera_index),
        width=width,
        height=height,
        fps=base_fps,
        bgr2rgb=bgr2rgb,
    )

    if aspect_ratio is None:
        aspect_ratio = width / height

    cam = RatioCropWrapper(cam, aspect_ratio)
    cam = ResizeWrapper(cam, (width, height))
    sensor = FrameSensor(cam)

    return sensor


def create_locomotion_actuator(
    osc_address: str = "127.0.0.1",
    osc_sender_port: int = 9000,
) -> LocomotionActuator:
    """Create LocomotionActuator object.

    Args:
        osc_address (str): IP address of VRChat client.
        osc_sender_port (int): Port of VRChat client.
    """
    controller = InputController(SimpleUDPClient(osc_address, osc_sender_port))
    controller = MultiInputWrapper(controller)
    controller = AxesLocomotionWrapper(controller)

    actuator = LocomotionActuator(controller)

    return actuator
