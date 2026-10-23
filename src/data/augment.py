"""Augmentation for the spatial candidate classifier.

Augmentation is only ever applied to **training** videos. The augmentation
policy is fixed a priori (see ``configs/baseline_kalman_hsv_cnn.yaml``) and is
never tuned on the validation or test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

__all__ = ["AugmentConfig", "augment_crop", "augment_batch"]


@dataclass
class AugmentConfig:
    """Parameters of the crop-level augmentation."""

    horizontal_flip: float = 0.5
    brightness_jitter: float = 0.20
    contrast_jitter: float = 0.20
    hue_jitter: float = 6.0
    saturation_jitter: float = 0.20
    gaussian_noise_std: float = 3.0
    gaussian_blur_prob: float = 0.15
    rotate_limit: float = 8.0
    scale_limit: float = 0.10
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizontal_flip": self.horizontal_flip,
            "brightness_jitter": self.brightness_jitter,
            "contrast_jitter": self.contrast_jitter,
            "hue_jitter": self.hue_jitter,
            "saturation_jitter": self.saturation_jitter,
            "gaussian_noise_std": self.gaussian_noise_std,
            "gaussian_blur_prob": self.gaussian_blur_prob,
            "rotate_limit": self.rotate_limit,
            "scale_limit": self.scale_limit,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AugmentConfig":
        cfg = cls()
        if not data:
            return cfg
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, type(getattr(cfg, key))(value))
        return cfg


def augment_crop(crop: np.ndarray, rng: np.random.Generator, config: AugmentConfig | None = None) -> np.ndarray:
    """Apply the augmentation policy to a single BGR crop."""
    cfg = config or AugmentConfig()
    if not cfg.enabled or crop.size == 0:
        return crop

    out = crop
    h, w = out.shape[:2]

    if cfg.horizontal_flip > 0 and rng.random() < cfg.horizontal_flip:
        out = cv2.flip(out, 1)

    if cfg.rotate_limit > 0:
        angle = float(rng.uniform(-cfg.rotate_limit, cfg.rotate_limit))
        scale = float(1.0 + rng.uniform(-cfg.scale_limit, cfg.scale_limit))
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if cfg.brightness_jitter > 0 or cfg.contrast_jitter > 0:
        alpha = 1.0 + float(rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
        beta = float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)) * 255.0
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if cfg.hue_jitter > 0 or cfg.saturation_jitter > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-cfg.hue_jitter, cfg.hue_jitter))) % 180
        hsv[..., 1] = np.clip(
            hsv[..., 1] * (1.0 + float(rng.uniform(-cfg.saturation_jitter, cfg.saturation_jitter))), 0, 255
        ).astype(np.int16)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if cfg.gaussian_blur_prob > 0 and rng.random() < cfg.gaussian_blur_prob:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if cfg.gaussian_noise_std > 0 and rng.random() < 0.5:
        noise = rng.normal(0.0, cfg.gaussian_noise_std, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def augment_batch(
    crops: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig | None = None,
) -> np.ndarray:
    """Apply the augmentation policy to a batch of BGR crops."""
    if crops.size == 0:
        return crops
    return np.stack([augment_crop(crop, rng, config) for crop in crops])
