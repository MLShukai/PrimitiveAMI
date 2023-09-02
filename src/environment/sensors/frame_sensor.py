import torch
from vrchat_io.abc.video_capture import VideoCapture

from .sensor import Sensor


class FrameSensor(Sensor):
    def __init__(self, camera: VideoCapture):
        self.camera = camera

    def read(self) -> torch.Tensor:
        return torch.from_numpy(self.camera.read()).clone()
