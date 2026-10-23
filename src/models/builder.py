"""Assemble a complete detector from the stored checkpoints.

This is the single place where the spatial model, the temporal model, the
calibrator, the feature standardiser and the alarm policy are wired
together, so that the training, calibration, evaluation and benchmarking
entry points all exercise exactly the same system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.calibration import Calibrator, build_calibrator
from ..models.calibrated_detector import (
    AlarmPolicyConfig,
    CalibratedDetector,
    DetectorConfig,
)
from ..models.candidate_features import FeatureScaler
from ..models.evidence_accumulator import EvidenceAccumulator, EvidenceConfig, ReliabilityModel
from ..models.fixed_temporal import TemporalVoter, VotingConfig
from ..models.spatial_cnn import SpatialCNN, build_spatial_cnn
from ..models.temporal_gru import GRUConfig, TemporalGRU
from ..models.temporal_tcn import TCNConfig, TemporalTCN
from ..utils.checkpoints import load_checkpoint
from ..utils.logging import get_logger

__all__ = ["build_detector", "DetectorBundle", "VARIANT_SPECS"]

LOGGER = get_logger("models.builder")

# The reference models of the benchmark. They are selected by name so that
# the evaluation scripts can switch between them without touching the
# checkpoints.
VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "spatial": {"temporal_kind": "none"},
    "voting_mean": {"temporal_kind": "voting", "voting": {"mode": "mean"}},
    "voting_median": {"temporal_kind": "voting", "voting": {"mode": "median"}},
    "voting_majority": {"temporal_kind": "voting", "voting": {"mode": "majority"}},
    "voting_max": {"temporal_kind": "voting", "voting": {"mode": "max"}},
    "voting_persistence": {"temporal_kind": "voting", "voting": {"mode": "persistence"}},
}


class DetectorBundle:
    """A detector plus the metadata needed to reproduce it."""

    def __init__(self, detector: CalibratedDetector, meta: dict[str, Any]) -> None:
        self.detector = detector
        self.meta = meta

    @property
    def temporal_kind(self) -> str:
        return str(self.meta.get("temporal_kind", "none"))


def build_detector(
    spatial_checkpoint: str | Path | None = None,
    temporal_checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    fps: float = 25.0,
    policy_overrides: dict[str, Any] | None = None,
    detector_config: DetectorConfig | None = None,
    video_id: str = "",
    temporal_kind_override: str | None = None,
    calibrator_override: str | None = None,
    variant: str | None = None,
) -> DetectorBundle:
    """Build a :class:`CalibratedDetector` from checkpoints.

    Args:
        spatial_checkpoint: Path to the spatial CNN checkpoint.
        temporal_checkpoint: Path to the temporal checkpoint (``gru``/``tcn``)
            or to the evidence/voting payload.
        device: Torch device.
        fps: Nominal frame rate of the stream.
        policy_overrides: Overrides for the alarm policy configuration.
        detector_config: Full detector configuration (defaults are derived
            from the checkpoint metadata).
        video_id: Video id used in the decision log.
        temporal_kind_override: Force a temporal module (used by the
            "CNN only" and "CNN plus fixed persistence" ablations).
        calibrator_override: Force a calibration method (``none`` disables the
            calibration, used by the "no calibration" ablation).
    """
    dev = torch.device(device)
    config = detector_config or DetectorConfig(fps=fps)

    spatial_model: SpatialCNN | None = None
    spatial_meta: dict[str, Any] = {}
    if spatial_checkpoint is not None:
        ckpt = load_checkpoint(spatial_checkpoint, map_location="cpu")
        spatial_meta = ckpt.meta
        spatial_model = build_spatial_cnn(spatial_meta.get("model_config", {}))
        spatial_model.load_state_dict(ckpt.model)
        spatial_model.to(dev).eval()
        input_size = spatial_meta.get("model_config", {}).get("input_size", [50, 50])
        config.spatial_input_size = (int(input_size[0]), int(input_size[1]))

    temporal_model: torch.nn.Module | None = None
    calibrator: Calibrator | None = None
    scaler: FeatureScaler | None = None
    voter: TemporalVoter | None = None
    evidence: EvidenceAccumulator | None = None
    temporal_kind = "none"
    temporal_meta: dict[str, Any] = {}

    if temporal_checkpoint is not None and str(temporal_checkpoint).lower() not in ("none", "null", ""):
        path = Path(temporal_checkpoint)
        if path.suffix == ".pt" and _is_state_dict_payload(path):
            ckpt = load_checkpoint(path, map_location="cpu")
            temporal_meta = ckpt.meta
            model_cfg = temporal_meta.get("model_config", {})
            kind = str(temporal_meta.get("model_type", config.temporal_kind)).lower()
            if kind == "gru":
                temporal_model = TemporalGRU(GRUConfig.from_dict(model_cfg))
            elif kind == "tcn":
                temporal_model = TemporalTCN(TCNConfig.from_dict(model_cfg))
            else:
                raise ValueError(f"unsupported temporal model type: {kind!r}")
            temporal_model.load_state_dict(ckpt.model)
            temporal_model.to(dev).eval()
            temporal_kind = kind
            config.sequence_length = int(temporal_meta.get("sequence_length", config.sequence_length))
            scaler = FeatureScaler.from_state_dict(temporal_meta.get("feature_scaler", {}))
            if temporal_meta.get("calibrator"):
                calibrator = build_calibrator(temporal_meta["calibrator"].get("method", "none"))
                calibrator = _restore_calibrator(calibrator, temporal_meta["calibrator"])
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if "evidence_config" in payload:
                ev_cfg = EvidenceConfig.from_dict(payload["evidence_config"])
                reliability = ReliabilityModel.from_state_dict(payload.get("reliability_model"))
                evidence = EvidenceAccumulator(ev_cfg, reliability)
                temporal_kind = "evidence"
                config.evidence = {**ev_cfg.to_dict(), "reliability_model": reliability.state_dict()}
            elif "voting_config" in payload:
                voting_cfg = VotingConfig.from_dict(payload["voting_config"])
                voter = TemporalVoter(voting_cfg, fps=fps)
                temporal_kind = "voting"
                config.voting = voting_cfg.to_dict()
            else:
                raise ValueError(f"unrecognised temporal payload: {path}")

    config.temporal_kind = temporal_kind_override or temporal_kind
    config.fps = fps

    if variant:
        spec = VARIANT_SPECS.get(variant)
        if spec is None:
            raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANT_SPECS)}")
        config.temporal_kind = str(spec["temporal_kind"])
        if config.temporal_kind == "voting":
            config.voting = {**VotingConfig().to_dict(), **(spec.get("voting") or {})}
            voter = TemporalVoter(VotingConfig.from_dict(config.voting), fps=fps)
        elif config.temporal_kind == "none":
            temporal_model = None
            evidence = None
        LOGGER.info("variant %s -> temporal kind %s", variant, config.temporal_kind)

    if calibrator_override is not None:
        calibrator = build_calibrator(calibrator_override)

    policy_config = AlarmPolicyConfig.from_dict(config.policy)
    if policy_overrides:
        merged = policy_config.to_dict()
        merged.update(policy_overrides)
        policy_config = AlarmPolicyConfig.from_dict(merged)

    detector = CalibratedDetector(
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        calibrator=calibrator,
        policy_config=policy_config,
        voter=voter,
        evidence=evidence,
        feature_scaler=scaler,
        config=config,
        device=dev,
        video_id=video_id,
    )

    meta = {
        "temporal_kind": temporal_kind,
        "spatial_meta": spatial_meta,
        "temporal_meta": temporal_meta,
        "policy": policy_config.to_dict(),
        "fps": fps,
    }
    return DetectorBundle(detector, meta)


def _is_state_dict_payload(path: Path) -> bool:
    """True when the file is a torch checkpoint produced by ``save_checkpoint``."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return "model" in payload and isinstance(payload.get("model"), dict)


def _restore_calibrator(calibrator: Calibrator, state: dict[str, Any]) -> Calibrator:
    """Restore the parameters of a fitted calibrator."""
    method = state.get("method", "none")
    if method == "temperature":
        calibrator.temperature = float(state.get("temperature", 1.0))
    elif method == "platt":
        calibrator.coef_ = np.asarray(state.get("coef_", [1.0]), dtype=np.float64)
        calibrator.intercept_ = float(state.get("intercept_", 0.0))
    elif method == "isotonic":
        calibrator.x_thresholds_ = np.asarray(state.get("x_thresholds_", [-np.inf, np.inf]), dtype=np.float64)
        calibrator.y_thresholds_ = np.asarray(state.get("y_thresholds_", [0.0, 1.0]), dtype=np.float64)
    return calibrator
