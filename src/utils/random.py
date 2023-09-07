import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seed to all random generator."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
