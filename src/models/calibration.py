"""Probability calibration.

Calibration is fitted **only** on the validation split and applied unchanged
to the test split. Four methods are supported so that the paper can compare
them (hypothesis H5 and ablation "temperature scaling versus isotonic"):

* ``none``      -- raw sigmoid output,
* ``temperature``-- single-parameter temperature scaling,
* ``platt``     -- logistic regression on the logit,
* ``isotonic``  -- non-parametric isotonic regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Calibrator",
    "NoCalibration",
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibration",
    "build_calibrator",
    "CALIBRATION_METHODS",
]

CALIBRATION_METHODS = ("none", "temperature", "platt", "isotonic")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class Calibrator:
    """Base class: maps logits to calibrated probabilities."""

    method: str = "none"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "Calibrator":
        raise NotImplementedError

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.method}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Calibrator":
        method = str(state.get("method", "none"))
        if method == "none":
            return NoCalibration()
        if method == "temperature":
            obj = TemperatureScaling()
            obj.temperature = float(state["temperature"])
            return obj
        if method == "platt":
            obj = PlattScaling()
            obj.coef_ = np.asarray(state["coef_"], dtype=np.float64)
            obj.intercept_ = float(state["intercept_"])
            return obj
        if method == "isotonic":
            obj = IsotonicCalibration()
            obj.x_thresholds_ = np.asarray(state["x_thresholds_"], dtype=np.float64)
            obj.y_thresholds_ = np.asarray(state["y_thresholds_"], dtype=np.float64)
            return obj
        raise ValueError(f"unknown calibration method: {method!r}")


class NoCalibration(Calibrator):
    """Identity calibration: the sigmoid of the logit."""

    method = "none"

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "NoCalibration":
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(logits, dtype=np.float64))


class TemperatureScaling(Calibrator):
    """Divide the logit by a positive temperature before the sigmoid."""

    method = "temperature"

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if logits.size == 0:
            return self

        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            p = _sigmoid(logits / t)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return _sigmoid(logits / max(self.temperature, 1e-6))

    def state_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature}


class PlattScaling(Calibrator):
    """Logistic regression ``sigmoid(a * logit + b)`` fitted by gradient descent."""

    method = "platt"

    def __init__(self) -> None:
        self.coef_: np.ndarray = np.array([1.0])
        self.intercept_: float = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        from sklearn.linear_model import LogisticRegression

        logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels).astype(int).reshape(-1)
        if logits.shape[0] < 2 or len(np.unique(labels)) < 2:
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits, labels)
        self.coef_ = model.coef_.reshape(-1).astype(np.float64)
        self.intercept_ = float(model.intercept_[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return _sigmoid(self.coef_[0] * logits + self.intercept_)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "coef_": self.coef_.tolist(),
            "intercept_": self.intercept_,
        }


class IsotonicCalibration(Calibrator):
    """Monotonic isotonic regression mapping logit -> probability."""

    method = "isotonic"

    def __init__(self) -> None:
        self.x_thresholds_: np.ndarray = np.array([-np.inf, np.inf])
        self.y_thresholds_: np.ndarray = np.array([0.0, 1.0])

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        from sklearn.isotonic import IsotonicRegression

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if logits.size < 2:
            return self
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(logits, labels)
        self.x_thresholds_ = np.asarray(iso.X_thresholds_, dtype=np.float64)
        self.y_thresholds_ = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return np.interp(logits, self.x_thresholds_, self.y_thresholds_)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "x_thresholds_": self.x_thresholds_.tolist(),
            "y_thresholds_": self.y_thresholds_.tolist(),
        }


def build_calibrator(method: str) -> Calibrator:
    """Instantiate a calibrator by name."""
    method = method.lower()
    if method == "none":
        return NoCalibration()
    if method in ("temperature", "temperature_scaling"):
        return TemperatureScaling()
    if method in ("platt", "platt_scaling"):
        return PlattScaling()
    if method in ("isotonic", "isotonic_regression"):
        return IsotonicCalibration()
    raise ValueError(
        f"unknown calibration method {method!r}; expected one of {CALIBRATION_METHODS}"
    )
