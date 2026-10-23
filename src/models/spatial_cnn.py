"""Compact spatial CNN for candidate crops (Model 1 of the specification).

The network is deliberately small so that it can run on edge hardware. It
operates on 50x50 RGB crops of Kalman/HSV candidates and outputs a single
logit per crop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

__all__ = ["SpatialCNNConfig", "SpatialCNN", "preprocess_crops", "build_spatial_cnn"]


@dataclass
class SpatialCNNConfig:
    """Architecture of the candidate classifier."""

    input_size: tuple[int, int] = (50, 50)
    channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernels: list[int] = field(default_factory=lambda: [3, 3, 3])
    activation: str = "leaky_relu"
    dropout: float = 0.50
    normalization: str = "batch_norm"
    num_outputs: int = 1
    in_channels: int = 3

    def to_dict(self) -> dict:
        return {
            "input_size": list(self.input_size),
            "channels": list(self.channels),
            "kernels": list(self.kernels),
            "activation": self.activation,
            "dropout": self.dropout,
            "normalization": self.normalization,
            "num_outputs": self.num_outputs,
            "in_channels": self.in_channels,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "SpatialCNNConfig":
        if not data:
            return cls()
        cfg = cls()
        if "input_size" in data:
            cfg.input_size = tuple(int(v) for v in data["input_size"])  # type: ignore[assignment]
        for key in ("channels", "kernels"):
            if key in data:
                setattr(cfg, key, [int(v) for v in data[key]])
        for key in ("activation", "normalization"):
            if key in data:
                setattr(cfg, key, str(data[key]))
        if "dropout" in data:
            cfg.dropout = float(data["dropout"])
        if "num_outputs" in data:
            cfg.num_outputs = int(data["num_outputs"])
        if "in_channels" in data:
            cfg.in_channels = int(data["in_channels"])
        return cfg


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "group_norm":
        return nn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels)
    if kind in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"unknown normalization: {kind!r}")


def _make_act(kind: str) -> nn.Module:
    if kind == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation: {kind!r}")


class SpatialCNN(nn.Module):
    """Small CNN over candidate crops producing a binary smoke probability."""

    def __init__(self, config: SpatialCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialCNNConfig()

        layers: list[nn.Module] = []
        in_c = self.config.in_channels
        h, w = self.config.input_size
        for out_c, k in zip(self.config.channels, self.config.kernels):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, padding=k // 2, bias=False))
            layers.append(_make_norm(self.config.normalization, out_c))
            layers.append(_make_act(self.config.activation))
            layers.append(nn.MaxPool2d(2))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, h, w)
            out_dim = int(np.prod(self.features(dummy).shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(out_dim, 128),
            _make_act(self.config.activation),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B,)`` logits."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) tensor, got {tuple(x.shape)}")
        return self.classifier(self.features(x)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return the smoke probability for a batch of crops."""
        return torch.sigmoid(self.forward(x))


def preprocess_crops(
    crops: np.ndarray | list[np.ndarray],
    input_size: tuple[int, int] = (50, 50),
) -> torch.Tensor:
    """Convert BGR uint8 crops into a normalised batch tensor.

    Args:
        crops: An ``(N, H, W, 3)`` uint8 BGR array or a list of such arrays.
        input_size: Target spatial size (height, width).

    Returns:
        A ``(N, 3, H, W)`` float32 tensor with RGB values scaled to [0, 1].
    """
    import cv2

    if isinstance(crops, list):
        crops = np.stack(crops) if crops else np.zeros((0, input_size[0], input_size[1], 3), np.uint8)
    if crops.size == 0:
        return torch.zeros((0, 3, input_size[0], input_size[1]), dtype=torch.float32)

    crops = np.asarray(crops)
    if crops.shape[1] != input_size[0] or crops.shape[2] != input_size[1]:
        resized = np.empty(
            (crops.shape[0], input_size[0], input_size[1], 3), dtype=np.uint8
        )
        for i, crop in enumerate(crops):
            resized[i] = cv2.resize(crop, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
        crops = resized

    rgb = crops[..., ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
    return tensor


def build_spatial_cnn(config: dict | SpatialCNNConfig | None = None) -> SpatialCNN:
    """Build a :class:`SpatialCNN` from a config dict."""
    if isinstance(config, SpatialCNNConfig):
        return SpatialCNN(config)
    return SpatialCNN(SpatialCNNConfig.from_dict(config))
