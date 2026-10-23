"""Probabilistic evidence accumulator (Model 5 of the specification).

A non-neural temporal model that integrates the log-odds of the calibrated
frame probability with a reliability weight and a forgetting factor:

.. math::

    l_t = \\log\\frac{p_t}{1 - p_t}, \\qquad
    E_t = \\lambda\\, E_{t-1} + r_t\\, l_t

Two thresholds with hysteresis decide the alarm state:

* ``alarm_threshold``  -- the accumulator is considered conclusive,
* ``cancel_threshold`` -- the accumulator is reset to idle.

``r_t`` (the reliability of the current observation) is a logistic function of
interpretable quality signals: motion consistency, candidate persistence,
blur, brightness, candidate size and the model's own uncertainty. The
coefficients are fitted on the *training* split; the thresholds are selected
on the *validation* split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EvidenceConfig",
    "ReliabilityModel",
    "EvidenceAccumulator",
    "reliability_features",
    "RELIABILITY_FEATURE_NAMES",
]

RELIABILITY_FEATURE_NAMES: list[str] = [
    "motion_consistency",
    "persistence",
    "blur_score",
    "brightness",
    "candidate_size",
    "uncertainty",
]


def reliability_features(
    feature_row: np.ndarray,
    uncertainty: float = 0.0,
    motion_consistency: float = 0.0,
) -> np.ndarray:
    """Build the reliability feature vector from a frame feature row.

    Args:
        feature_row: The 16-dimensional frame feature vector.
        uncertainty: Predicted uncertainty of the temporal/spatial model.
        motion_consistency: Correlation-like measure of the motion signal in
            the recent window, in [0, 1].

    Returns:
        A ``(len(RELIABILITY_FEATURE_NAMES),)`` float32 vector.
    """
    from .candidate_features import FEATURE_INDEX

    def get(name: str) -> float:
        idx = FEATURE_INDEX.get(name)
        if idx is None or idx >= len(feature_row):
            return 0.0
        return float(feature_row[idx])

    return np.array(
        [
            float(np.clip(motion_consistency, 0.0, 1.0)),
            get("persistence"),
            get("blur_score"),
            get("brightness"),
            get("area_ratio"),
            float(uncertainty),
        ],
        dtype=np.float32,
    )


@dataclass
class ReliabilityModel:
    """Logistic model mapping quality signals to a reliability weight in [0, 1]."""

    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the reliability weight for one or many feature vectors."""
        feats = np.atleast_2d(np.asarray(features, dtype=np.float64))
        scores = np.full(feats.shape[0], self.bias, dtype=np.float64)
        for i, name in enumerate(RELIABILITY_FEATURE_NAMES):
            w = float(self.weights.get(name, 0.0))
            if w != 0.0:
                scores += w * feats[:, i]
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ReliabilityModel":
        """Fit the logistic reliability model (positive = reliable observation)."""
        from sklearn.linear_model import LogisticRegression

        feats = np.atleast_2d(np.asarray(features, dtype=np.float64))
        y = np.asarray(labels).astype(int).reshape(-1)
        if feats.shape[0] < 4 or len(np.unique(y)) < 2:
            self.fitted = True
            return self

        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(feats, y, sample_weight=sample_weight)
        self.bias = float(model.intercept_[0])
        self.weights = {
            name: float(coef) for name, coef in zip(RELIABILITY_FEATURE_NAMES, model.coef_[0])
        }
        self.fitted = True
        return self

    def state_dict(self) -> dict:
        return {"bias": self.bias, "weights": dict(self.weights), "fitted": self.fitted}

    @classmethod
    def from_state_dict(cls, state: dict | None) -> "ReliabilityModel":
        state = state or {}
        return cls(
            bias=float(state.get("bias", 0.0)),
            weights={k: float(v) for k, v in (state.get("weights") or {}).items()},
            fitted=bool(state.get("fitted", False)),
        )


@dataclass
class EvidenceConfig:
    """Configuration of the evidence accumulator."""

    alarm_threshold: float = 8.0
    cancel_threshold: float = -4.0
    forgetting_factor: float = 0.95
    min_reliability: float = 0.05
    max_reliability: float = 1.0
    clip_logit: float = 12.0
    use_reliability_model: bool = True

    def to_dict(self) -> dict:
        return {
            "alarm_threshold": self.alarm_threshold,
            "cancel_threshold": self.cancel_threshold,
            "forgetting_factor": self.forgetting_factor,
            "min_reliability": self.min_reliability,
            "max_reliability": self.max_reliability,
            "clip_logit": self.clip_logit,
            "use_reliability_model": self.use_reliability_model,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "EvidenceConfig":
        cfg = cls()
        for key in (
            "alarm_threshold",
            "cancel_threshold",
            "forgetting_factor",
            "min_reliability",
            "max_reliability",
            "clip_logit",
        ):
            if data and key in data:
                setattr(cfg, key, float(data[key]))
        if data and "use_reliability_model" in data:
            cfg.use_reliability_model = bool(data["use_reliability_model"])
        return cfg


class EvidenceAccumulator:
    """Stateful leaky integrator of calibrated log-odds evidence."""

    def __init__(
        self,
        config: EvidenceConfig | None = None,
        reliability_model: ReliabilityModel | None = None,
    ) -> None:
        self.config = config or EvidenceConfig()
        self.reliability_model = reliability_model or ReliabilityModel()
        self.evidence: float = 0.0
        self.state: str = "idle"
        self.peak_evidence: float = 0.0
        self._alarm_frames: int = 0

    def reset(self) -> None:
        """Clear the accumulator (called at the start of each video)."""
        self.evidence = 0.0
        self.state = "idle"
        self.peak_evidence = 0.0
        self._alarm_frames = 0

    def step(
        self,
        probability: float,
        reliability: float | None = None,
        reliability_features: np.ndarray | None = None,
    ) -> dict[str, float | str]:
        """Integrate one frame of evidence.

        Args:
            probability: Calibrated smoke probability of the current frame.
            reliability: Explicit reliability weight; if ``None`` it is derived
                from ``reliability_features`` via the reliability model.
            reliability_features: Feature vector for the reliability model.

        Returns:
            A dictionary with the updated ``evidence``, the ``reliability``
            weight, the ``state`` and a boolean ``alarm``.
        """
        p = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = float(np.log(p / (1.0 - p)))
        logit = float(np.clip(logit, -self.config.clip_logit, self.config.clip_logit))

        if reliability is None:
            if reliability_features is not None and self.config.use_reliability_model:
                reliability = float(self.reliability_model.predict(reliability_features)[0])
            else:
                reliability = 1.0
        reliability = float(np.clip(reliability, self.config.min_reliability, self.config.max_reliability))

        self.evidence = self.config.forgetting_factor * self.evidence + reliability * logit
        self.peak_evidence = max(self.peak_evidence, self.evidence)

        if self.state == "idle":
            if self.evidence >= self.config.alarm_threshold:
                self.state = "alarm"
                self._alarm_frames = 1
        else:
            if self.evidence <= self.config.cancel_threshold:
                self.state = "idle"
                self._alarm_frames = 0
            else:
                self._alarm_frames += 1

        return {
            "evidence": self.evidence,
            "reliability": reliability,
            "state": self.state,
            "alarm": self.state == "alarm",
            "logit": logit,
        }
