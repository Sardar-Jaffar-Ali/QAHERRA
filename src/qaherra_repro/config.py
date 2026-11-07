"""Configuration loading and deterministic experiment setup."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping configuration files."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_path(paths: dict[str, Any], key: str) -> Path:
    """Resolve one configured path and fail early with a useful error."""
    if key not in paths:
        raise KeyError(f"Missing path setting: {key}")
    path = Path(paths[key]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured path does not exist: {key}={path}")
    return path

