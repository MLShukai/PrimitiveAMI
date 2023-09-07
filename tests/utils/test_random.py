import os
import random

import numpy as np
import pytest
import torch

from src.utils.random import seed_everything


def test_seed_python_stdlib():
    seed_everything(9)
    val1 = random.random()
    val2 = random.random()

    seed_everything(9)
    val1_new = random.random()
    val2_new = random.random()

    assert val1 == val1_new
    assert val2 == val2_new


def test_seed_numpy():
    seed_everything(9)
    val1 = np.random.rand()
    val2 = np.random.rand()

    seed_everything(9)
    val1_new = np.random.rand()
    val2_new = np.random.rand()

    assert val1 == val1_new
    assert val2 == val2_new


def test_seed_torch_cpu():
    seed_everything(9)
    val1 = torch.rand(1)
    val2 = torch.rand(1)

    seed_everything(9)
    val1_new = torch.rand(1)
    val2_new = torch.rand(1)

    assert torch.equal(val1, val1_new)
    assert torch.equal(val2, val2_new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Not CUDA Available.")
def test_seed_torch_cuda():
    seed_everything(9)
    val1 = torch.rand(1, device="cuda")
    val2 = torch.rand(1, device="cuda")

    seed_everything(9)
    val1_new = torch.rand(1, device="cuda")
    val2_new = torch.rand(1, device="cuda")

    assert torch.equal(val1, val1_new)
    assert torch.equal(val2, val2_new)


def test_python_hash_seed_set():
    seed_everything(9)
    assert os.environ["PYTHONHASHSEED"] == "9"
