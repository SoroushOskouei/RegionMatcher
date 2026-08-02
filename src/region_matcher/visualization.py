from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .preprocessing import read_rgb


def save_heatmap_visualization(
    image_path: str | Path,
    similarity_map: np.ndarray,
    output_path: str | Path,
) -> tuple[float, float]:
    """Overlay a similarity heatmap and mark its peak location."""
    rgb = read_rgb(image_path)
    height, width = rgb.shape[:2]

    resized_heatmap = cv2.resize(similarity_map, (width, height), interpolation=cv2.INTER_CUBIC)
    resized_heatmap = resized_heatmap - resized_heatmap.min()
    resized_heatmap = resized_heatmap / max(float(resized_heatmap.max()), 1e-8)
    heatmap_u8 = np.clip(resized_heatmap * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = np.clip(0.62 * rgb + 0.38 * colored_rgb, 0, 255).astype(np.uint8)

    peak_y, peak_x = np.unravel_index(np.argmax(similarity_map), similarity_map.shape)
    x_normalized = float((peak_x + 0.5) / similarity_map.shape[1])
    y_normalized = float((peak_y + 0.5) / similarity_map.shape[0])
    point_x = int(x_normalized * width)
    point_y = int(y_normalized * height)
    cv2.drawMarker(
        blended,
        (point_x, point_y),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(16, min(height, width) // 18),
        thickness=3,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)
    return x_normalized, y_normalized
