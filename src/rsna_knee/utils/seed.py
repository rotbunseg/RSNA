"""Reproducibility helper. Call once at the top of every training script."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic-ish cuDNN. We do NOT force fully deterministic algorithms
    # (torch.use_deterministic_algorithms(True)) because it disables some fast
    # cuDNN kernels and we are time-constrained on Kaggle GPU quota — a small
    # amount of run-to-run noise is an acceptable trade for speed here.
    torch.backends.cudnn.benchmark = True
