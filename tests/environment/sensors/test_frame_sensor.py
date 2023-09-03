import cv2
import numpy as np
import pytest
from pytest_mock import MockerFixture
from vrchat_io.vision import OpenCVVideoCapture

from src.environment.sensors.frame_sensor import FrameSensor

frame_shapes = ((480, 640, 3), (640, 480, 3), (480, 480, 3))


class TestFrameSensor:
    def mock_camera(self, mocker: MockerFixture, shape):
        mock = mocker.MagicMock(spec=OpenCVVideoCapture)
        mock.read.return_value = np.random.randint(0, 255, shape).astype(np.uint8)
        return mock

    @pytest.mark.parametrize("shape", frame_shapes)
    def test__init__(self, mocker, shape):
        mock_camera = self.mock_camera(mocker, shape)
        sensor = FrameSensor(camera=mock_camera)
        assert sensor.camera is mock_camera

    @pytest.mark.parametrize("shape", frame_shapes)
    def test_read(self, mocker, shape):
        sensor = FrameSensor(camera=self.mock_camera(mocker, shape))
        assert sensor.read().shape == (shape[-1], shape[0], shape[1])
