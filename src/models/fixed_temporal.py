"""Fixed temporal rules (Model 2 of the specification).

A GRU or TCN is only interesting if it beats *simple* temporal aggregation.
This module implements the classical alternatives that operate purely on the
stream of frame probabilities:

* ``mean``      -- moving average over a window,
* ``median``    -- moving median over a window,
* ``majority``  -- fraction of window frames above the threshold,
* ``max``       -- moving maximum over a window,
* ``persistence`` -- a frame is positive only when the threshold has been
  exceeded for at least ``persistence_seconds`` continuously.

All of them are causal (they only look at the past) and therefore usable
online.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = ["VotingMode", "TemporalVoter", "VOTING_MODES"]

VOTING_MODES = ("mean", "median", "majority", "max", "persistence")


@dataclass
class VotingConfig:
    """Configuration of a fixed temporal rule."""

    mode: str = "mean"
    window_seconds: float = 3.0
    confidence_threshold: float = 0.70
    persistence_seconds: float = 1.50

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "window_seconds": self.window_seconds,
            "confidence_threshold": self.confidence_threshold,
            "persistence_seconds": self.persistence_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "VotingConfig":
        cfg = cls()
        if not data:
            return cfg
        for key in ("mode",):
            if key in data:
                cfg.mode = str(data[key])
        for key in ("window_seconds", "confidence_threshold", "persistence_seconds"):
            if key in data:
                setattr(cfg, key, float(data[key]))
        return cfg


class TemporalVoter:
    """Causal temporal aggregation of frame probabilities."""

    def __init__(self, config: VotingConfig | None = None, fps: float = 25.0) -> None:
        self.config = config or VotingConfig()
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.window = max(1, int(round(self.config.window_seconds * self.fps)))
        self._buffer: deque[float] = deque(maxlen=self.window)
        self._streak: int = 0
        self._required_streak: int = max(1, int(round(self.config.persistence_seconds * self.fps)))

    def reset(self) -> None:
        """Clear the temporal state."""
        self._buffer.clear()
        self._streak = 0

    def step(self, probability: float) -> dict[str, float | bool]:
        """Aggregate the current frame probability.

        Returns:
            ``{"probability", "decision", "support", "streak"}`` where
            ``probability`` is the aggregated value, ``decision`` is the
            boolean alarm decision of this rule and ``support`` is the
            fraction of the window above the threshold.
        """
        p = float(np.clip(probability, 0.0, 1.0))
        self._buffer.append(p)

        values = np.asarray(self._buffer, dtype=np.float64)
        mode = self.config.mode

        if mode == "mean":
            agg = float(values.mean())
        elif mode == "median":
            agg = float(np.median(values))
        elif mode == "max":
            agg = float(values.max())
        elif mode == "majority":
            agg = float((values >= self.config.confidence_threshold).mean())
        elif mode == "persistence":
            agg = p
        else:
            raise ValueError(f"unknown voting mode: {mode!r}; expected one of {VOTING_MODES}")

        support = float((values >= self.config.confidence_threshold).mean())

        if mode == "persistence":
            if p >= self.config.confidence_threshold:
                self._streak += 1
            else:
                self._streak = 0
            decision = self._streak >= self._required_streak
        else:
            decision = agg >= self.config.confidence_threshold

        return {
            "probability": agg,
            "decision": bool(decision),
            "support": support,
            "streak": float(self._streak),
        }
