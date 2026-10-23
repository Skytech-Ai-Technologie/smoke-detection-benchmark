"""Compact causal TCN temporal model (Model 4 of the specification)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .temporal_base import TemporalHead, causal_pad

__all__ = ["TCNConfig", "TemporalTCN", "build_temporal_tcn"]


@dataclass
class TCNConfig:
    """Architecture of the temporal convolutional network."""

    sequence_length: int = 16
    input_features: int = 16
    channels: list[int] = field(default_factory=lambda: [16, 16, 16])
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    dropout: float = 0.10
    causal: bool = True
    horizon: int = 8
    head_dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "input_features": self.input_features,
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "dropout": self.dropout,
            "causal": self.causal,
            "horizon": self.horizon,
            "head_dropout": self.head_dropout,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TCNConfig":
        if not data:
            return cls()
        cfg = cls()
        for key in ("sequence_length", "input_features", "kernel_size", "horizon"):
            if key in data:
                setattr(cfg, key, int(data[key]))
        for key in ("channels", "dilations"):
            if key in data:
                setattr(cfg, key, [int(v) for v in data[key]])
        for key in ("dropout", "head_dropout"):
            if key in data:
                setattr(cfg, key, float(data[key]))
        if "causal" in data:
            cfg.causal = bool(data["causal"])
        return cfg


class TemporalTCN(nn.Module):
    """Dilated causal convolution network over a window of per-frame features.

    Every convolution is left-padded with its own dilation factor so the
    prediction for time ``t`` only depends on frames ``<= t``; this is
    required for online deployment and makes the streaming and offline
    evaluations equivalent.
    """

    def __init__(self, config: TCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()

        self.input_norm = nn.LayerNorm(self.config.input_features)

        n_layers = max(len(self.config.channels), len(self.config.dilations))
        channels = list(self.config.channels) + [self.config.channels[-1]] * (
            n_layers - len(self.config.channels)
        )
        dilations = list(self.config.dilations) + [self.config.dilations[-1]] * (
            n_layers - len(self.config.dilations)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dilations: list[int] = []
        in_c = self.config.input_features
        for out_c, dilation in zip(channels, dilations):
            self.convs.append(
                nn.Conv1d(
                    in_c,
                    out_c,
                    kernel_size=self.config.kernel_size,
                    dilation=dilation,
                    padding=0,
                )
            )
            self.norms.append(nn.BatchNorm1d(out_c))
            self.dilations.append(dilation)
            in_c = out_c

        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
        self.head = TemporalHead(in_c, dropout=self.config.head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``(B, T, F)`` -> dictionary of logits."""
        if x.ndim != 3:
            raise ValueError(f"expected a (B, T, F) tensor, got {tuple(x.shape)}")
        h = self.input_norm(x).transpose(1, 2)  # (B, F, T)
        k = self.config.kernel_size
        for conv, norm, dilation in zip(self.convs, self.norms, self.dilations):
            if self.config.causal:
                h = causal_pad(h, k, dilation)
            else:
                h = torch.nn.functional.pad(h, (k // 2, k // 2))
            h = self.dropout(norm(conv(h)))
        return self.head(h[..., -1])


def build_temporal_tcn(config: dict | TCNConfig | None = None) -> TemporalTCN:
    """Build a :class:`TemporalTCN` from a config dict."""
    if isinstance(config, TCNConfig):
        return TemporalTCN(config)
    return TemporalTCN(TCNConfig.from_dict(config))
