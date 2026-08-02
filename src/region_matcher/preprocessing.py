from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

from .constants import IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image directly to a square."""
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def letterbox_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize while preserving aspect ratio, then reflect-pad to a square."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image")

    scale = size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Normalize an RGB image with ImageNet statistics and convert to CHW."""
    image_f = image.astype(np.float32) / 255.0
    image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_f).permute(2, 0, 1).contiguous()


def preprocess_query(path: str | Path, query_size: int) -> torch.Tensor:
    return image_to_tensor(letterbox_square(read_rgb(path), query_size))


def preprocess_full(path: str | Path, full_size: int) -> torch.Tensor:
    return image_to_tensor(resize_square(read_rgb(path), full_size))
