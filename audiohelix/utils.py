from __future__ import annotations

import logging
import os
import time
from hashlib import md5

import numpy as np


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize; returns all-ones if every value is equal."""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val


def setup_logging(log_file: str) -> None:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True,
    )


def log(msg: str) -> None:
    """Writes to both stdout and the log file set up by setup_logging()."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    logging.getLogger().info(msg)
