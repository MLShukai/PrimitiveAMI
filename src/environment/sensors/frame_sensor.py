import torch
from vrchat_io.abc.video_capture import VideoCapture

from .sensor import Sensor


class FrameSensor(Sensor):
    def __init__(self, camera: VideoCapture):
        self.camera = camera

    def read(self) -> torch.Tensor:
        """Receive observed data and convert dtype and shape of array.

        Returns:
            torch.Tensor: Tensor formatted for torch DL models.
        """
        return torch.from_numpy(self.camera.read()).clone().permute(2, 0, 1) / 256.0
