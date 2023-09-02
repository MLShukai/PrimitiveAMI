from typing import Optional

import cv2
import torch
from vrchat_io.abc.video_capture import VideoCapture
from vrchat_io.vision import OpenCVVideoCapture
from vrchat_io.vision.wrappers import RatioCropWrapper, ResizeWrapper

from .sensor import Sensor


class FrameSensor(Sensor):
    def __init__(self, camera: VideoCapture):
        self.camera = camera

    def read(self) -> torch.Tensor:
        return torch.from_numpy(self.camera.read()).clone()
