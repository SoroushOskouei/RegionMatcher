from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .constants import IMAGE_EXTENSIONS


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve an explicit device or automatically select CUDA when available."""
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def list_images(path: str | Path) -> list[Path]:
    """Return one image or all supported images under a directory recursively."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gallery path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported gallery image type: {path.suffix}")
        return [path]

    images = sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No supported images found under: {path}")
    return images
