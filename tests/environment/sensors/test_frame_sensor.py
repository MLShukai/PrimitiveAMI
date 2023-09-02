import pytest
import torch
from src.environment.sensors.frame_sensor import FrameSensor

params_order = "width, height, fps, bgr2rgb, num_trials_on_read_failure, ratio, anchor, size, expected_shape"
params = (
    (640, 480, 30.0, True, 10, None, "center", None, (640, 480)),
    # width/height>ratio -> crop width(640 -> 480)
    (640, 480, 30.0, True, 10, 1.0, "center", None, (480, 480)),
    # width/heigth<ratio -> crop height(480->320)
    (640, 480, 30.0, True, 10, 2.0, "center", None, ((640, 320))),
    # crop by size
    (640, 480, 30.0, True, 10, None, "center", (500, 500), (500, 500))
)

class TestFrameSensor:

    @pytest.mark.parametrize(params_order, params)
    def test__init__(self, width, height, fps, bgr2rgb, num_trials_on_read_failure, ratio, anchor, size, expected_shape):
        mod = FrameSensor(0, width, height, fps, bgr2rgb, num_trials_on_read_failure, ratio, anchor, size)
        assert mod.camera.width == expected_shape[0]
        assert mod.camera.height == expected_shape[1]
        assert mod.camera.fps == fps
        assert mod.camera.bgr2rgb == bgr2rgb
        assert mod.camera.num_trials_on_read_failure == num_trials_on_read_failure
        assert mod.camera.anchor == anchor
    
    @pytest.mark.parametrize(params_order, params)
    def test_read(self, mocker, width, height, fps, bgr2rgb, num_trials_on_read_failure, ratio, anchor, size, expected_shape):
        with mocker.patch("cv2.VideoCapture") as mock:
            mock.return_value = torch.zeros((640, 480))
            sensor = FrameSensor(
                camera_index=0,
                width=width,
                height=height,
                fps=fps,
                bgr2rgb=bgr2rgb,
                num_trials_on_read_failure=num_trials_on_read_failure,
                ratio=ratio,
                anchor=anchor,
                size=size
            )
            assert sensor.read().shape == expected_shape
            
