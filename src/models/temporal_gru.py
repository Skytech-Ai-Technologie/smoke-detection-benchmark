"""Compact GRU temporal model (Model 3 of the specification)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .temporal_base import TemporalHead

__all__ = ["GRUConfig", "TemporalGRU", "build_temporal_gru"]


@dataclass
class GRUConfig:
    """Architecture of the GRU detector."""

    sequence_length: int = 16
    feature_dimension: int = 16
    hidden_dimension: int = 32
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    horizon: int = 8
    head_dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "feature_dimension": self.feature_dimension,
            "hidden_dimension": self.hidden_dimension,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "horizon": self.horizon,
            "head_dropout": self.head_dropout,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GRUConfig":
        if not data:
            return cls()
        cfg = cls()
        for key in (
            "sequence_length",
            "feature_dimension",
            "hidden_dimension",
            "num_layers",
            "horizon",
        ):
            if key in data:
                setattr(cfg, key, int(data[key]))
        for key in ("dropout", "head_dropout"):
            if key in data:
                setattr(cfg, key, float(data[key]))
        if "bidirectional" in data:
            cfg.bidirectional = bool(data["bidirectional"])
        return cfg


class TemporalGRU(nn.Module):
    """GRU over a window of per-frame features."""

    def __init__(self, config: GRUConfig | None = None) -> None:
        super().__init__()
        self.config = config or GRUConfig()

        self.input_norm = nn.LayerNorm(self.config.feature_dimension)
        self.gru = nn.GRU(
            input_size=self.config.feature_dimension,
            hidden_size=self.config.hidden_dimension,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        out_dim = self.config.hidden_dimension * (2 if self.config.bidirectional else 1)
        self.head = TemporalHead(out_dim, dropout=self.config.head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``(B, T, F)`` -> dictionary of logits.

        Only the representation of the *last* time step is used for the
        decision, which makes the model usable in a streaming setting.
        """
        if x.ndim != 3:
            raise ValueError(f"expected a (B, T, F) tensor, got {tuple(x.shape)}")
        h = self.input_norm(x)
        out, _ = self.gru(h)
        return self.head(out[:, -1, :])
