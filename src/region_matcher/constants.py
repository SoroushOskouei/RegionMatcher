from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CHECKPOINT = Path("models/model.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/match_results")
