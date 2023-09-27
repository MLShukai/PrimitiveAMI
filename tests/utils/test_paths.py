import os
from pathlib import Path

from src.utils import paths


def test_project_root():
    assert paths.PROJECT_ROOT == Path(os.path.abspath(__file__)).parent.parent.parent
