from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .constants import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR
from .modeling import load_checkpoint, soft_spatial_max
from .preprocessing import preprocess_full, preprocess_query
from .utils import list_images, resolve_device, seed_everything
from .visualization import save_heatmap_visualization


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    query: str | Path
    gallery: str | Path
    checkpoint: str | Path = DEFAULT_CHECKPOINT
    output: str | Path = DEFAULT_OUTPUT_DIR
    top_k: int = 10
    batch_size: int = 16
    full_size: int | None = None
    query_size: int | None = None
    device: str | None = None
    seed: int = 42


@dataclass(frozen=True, slots=True)
class MatchResult:
    path: str
    score: float
    x_normalized: float
    y_normalized: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_inference(options: InferenceOptions, *, show_progress: bool = True) -> list[MatchResult]:
    """Match a query crop against one image or a recursive gallery directory."""
    if options.top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if options.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    seed_everything(options.seed)
    device = resolve_device(options.device)
    print(f"Device: {device}")

    model, checkpoint_config = load_checkpoint(options.checkpoint, device)
    full_size = int(options.full_size or checkpoint_config.get("full_size", 384))
    query_size = int(options.query_size or checkpoint_config.get("query_size", 192))
    gallery_paths = list_images(options.gallery)

    query_tensor = preprocess_query(options.query, query_size).unsqueeze(0).to(device)
    with torch.inference_mode():
        query_embedding = model.encode_query(query_tensor)[0]

    records: list[MatchResult] = []
    best_score = -float("inf")
    best_path: Path | None = None
    best_map: np.ndarray | None = None

    starts = range(0, len(gallery_paths), options.batch_size)
    iterator = tqdm(starts, desc="Matching", disable=not show_progress)
    for start in iterator:
        batch_paths = gallery_paths[start : start + options.batch_size]
        batch = torch.stack([preprocess_full(path, full_size) for path in batch_paths]).to(device)

        with torch.inference_mode():
            full_maps = model.encode_full(batch)
            similarity = torch.einsum("d,bdhw->bhw", query_embedding, full_maps)
            scores = soft_spatial_max(similarity)

        maps_np = similarity.float().cpu().numpy()
        scores_np = scores.float().cpu().numpy()
        for path, score, heatmap in zip(batch_paths, scores_np, maps_np, strict=True):
            peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            result = MatchResult(
                path=str(path.resolve()),
                score=float(score),
                x_normalized=float((peak_x + 0.5) / heatmap.shape[1]),
                y_normalized=float((peak_y + 0.5) / heatmap.shape[0]),
            )
            records.append(result)

            if result.score > best_score:
                best_score = result.score
                best_path = path
                best_map = heatmap.copy()

    selected = sorted(records, key=lambda item: item.score, reverse=True)[: options.top_k]
    output_dir = Path(options.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_path = output_dir / "matches.json"
    with matches_path.open("w", encoding="utf-8") as file:
        json.dump([result.to_dict() for result in selected], file, indent=2)

    if best_path is not None and best_map is not None:
        save_heatmap_visualization(
            best_path,
            best_map,
            output_dir / "best_match_heatmap.jpg",
        )

    print(json.dumps([result.to_dict() for result in selected], indent=2))
    print(f"Saved: {matches_path}")
    if best_path is not None:
        print(f"Best match: {best_path}")
        print(f"Heatmap: {output_dir / 'best_match_heatmap.jpg'}")

    return selected
