import numpy as np
import pytest
from pytest_mock import MockerFixture

from src.environment.actuators.locomotion_actuator import LocomotionActuator
from src.environment.sensors.frame_sensor import FrameSensor
from src.utils.environment import create_frame_sensor, create_locomotion_actuator


@pytest.mark.parametrize(
    "camera_index,width,height,base_fps,bgr2rgb,aspect_ratio",
    [(0, 256, 256, 30.0, False, None), (1, 256, 128, 60.0, True, 1.5)],
)
def test_create_frame_sensor(mocker: MockerFixture, camera_index, width, height, base_fps, bgr2rgb, aspect_ratio):
    mock_capture = mocker.patch("cv2.VideoCapture")()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, image)

    sensor = create_frame_sensor(camera_index, width, height, base_fps, bgr2rgb, aspect_ratio)

    assert isinstance(sensor, FrameSensor)
    assert sensor.read().shape == (3, height, width)


def test_create_locomotion_actuator(mocker: MockerFixture):
    mock_client = mocker.patch("pythonosc.udp_client.SimpleUDPClient")

    actuator = create_locomotion_actuator("127.0.0.1", 9000)

    assert isinstance(actuator, LocomotionActuator)
