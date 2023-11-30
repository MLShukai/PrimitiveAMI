import hydra
from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from src.utils.paths import PROJECT_ROOT


def test_frame_locomotion_sleep(mocker: MockerFixture):
    mocker.patch("cv2.VideoCapture")  # Avoid no camera device error
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/environment/frame_locomotion_sleep.yaml")
    hydra.utils.instantiate(cfg)


def test_frame_discrete_sleep(mocker: MockerFixture):
    mocker.patch("cv2.VideoCapture")  # Avoid no camera device error
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/environment/frame_discrete_sleep.yaml")
    hydra.utils.instantiate(cfg)
