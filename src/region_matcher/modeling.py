from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class GeM(nn.Module):
    """Generalized-mean pooling with a learnable exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=8.0)
        x = x.clamp(min=self.eps).pow(p)
        return F.adaptive_avg_pool2d(x, output_size=1).pow(1.0 / p).flatten(1)


class DenseResNetFPN(nn.Module):
    """Stride-8 feature grid combining ResNet-50 layer2 and layer3."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.project2 = nn.Conv2d(512, embedding_dim, kernel_size=1, bias=False)
        self.project3 = nn.Conv2d(1024, embedding_dim, kernel_size=1, bias=False)
        self.smooth = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        feature2 = self.layer2(x)
        feature3 = self.layer3(feature2)
        pyramid = self.project2(feature2) + F.interpolate(
            self.project3(feature3),
            size=feature2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.smooth(pyramid)


class RegionMatcher(nn.Module):
    """Encode query crops and gallery images into a shared embedding space."""

    def __init__(self, embedding_dim: int = 256, query_rotations: int = 4):
        super().__init__()
        if query_rotations not in (1, 2, 4):
            raise ValueError("query_rotations must be one of: 1, 2, 4")
        self.query_rotations = query_rotations
        self.encoder = DenseResNetFPN(embedding_dim=embedding_dim)
        self.pool = GeM()

    def encode_full(self, full: torch.Tensor) -> torch.Tensor:
        dense = F.relu(self.encoder(full))
        return F.normalize(dense, dim=1)

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        rotations = [torch.rot90(query, k=k, dims=(-2, -1)) for k in range(self.query_rotations)]
        stacked = torch.cat(rotations, dim=0)
        dense = self.encoder(stacked)
        pooled = F.normalize(self.pool(F.relu(dense)), dim=1)
        batch_size = query.shape[0]
        pooled = pooled.view(self.query_rotations, batch_size, -1).mean(dim=0)
        return F.normalize(pooled, dim=1)


def soft_spatial_max(similarity: torch.Tensor, temperature: float = 0.04) -> torch.Tensor:
    """Smoothly aggregate a spatial similarity map into one score per image."""
    flat = similarity.flatten(start_dim=-2)
    count = flat.shape[-1]
    return temperature * (torch.logsumexp(flat / temperature, dim=-1) - math.log(count))


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[RegionMatcher, dict[str, Any]]:
    """Load a training-script checkpoint and construct the matching model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Place your model at models/model.pt or pass --checkpoint."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint is not in the expected training-script format")

    config = checkpoint.get("config", {})
    model = RegionMatcher(
        embedding_dim=int(config.get("embedding_dim", 256)),
        query_rotations=int(config.get("query_rotations", 4)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config
