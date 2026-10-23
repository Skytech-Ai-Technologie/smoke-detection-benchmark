"""Shared pieces of the temporal model family.

Every temporal model in this project maps a ``(B, T, F)`` sequence of frame
features to three quantities:

* ``now``   -- logit that smoke is present in the current frame,
* ``horizon`` -- logit that smoke will be present within the next
  ``horizon`` frames (an early-warning head),
* ``uncertainty`` -- a non-negative log-variance estimate used by the adaptive
  alarm policy.

Causal (left-padded) convolutions are used by the TCN so that the detector
can run online without access to future frames.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TemporalHead", "TemporalOutput", "causal_pad"]


class TemporalHead(nn.Module):
    """Three-headed output layer: current, horizon and uncertainty."""

    def __init__(self, in_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.now = nn.Linear(in_features, 1)
        self.horizon = nn.Linear(in_features, 1)
        self.uncertainty = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.dropout(features)
        return {
            "now": self.now(h).squeeze(-1),
            "horizon": self.horizon(h).squeeze(-1),
            "log_variance": self.uncertainty(h).squeeze(-1),
        }


class TemporalOutput(dict):
    """Dictionary subclass so outputs can be used like a mapping."""


def causal_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """Left-pad ``x`` so that a convolution of the given kernel stays causal.

    Args:
        x: ``(B, C, T)`` tensor.
        kernel_size: Convolution kernel size.
        dilation: Convolution dilation.

    Returns:
        The left-padded tensor.
    """
    total = (kernel_size - 1) * dilation
    return torch.nn.functional.pad(x, (total, 0))
