"""Calibrated adaptive detector (Model 6 of the specification).

This is the proposed end-to-end system:

    Kalman background model
        -> HSV candidate extraction
        -> compact spatial CNN
        -> calibrated frame probability
        -> temporal module (causal TCN / GRU / evidence accumulator / fixed rule)
        -> adaptive alarm policy
        -> interpretable decision log

The same code path serves the offline frame evaluation, the offline video
evaluation and the online streaming evaluation, which guarantees that the
three modes cannot silently disagree.

Every decision is logged with the quantities required by the specification so
that an alarm can be explained after the fact.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from .candidate_features import FEATURE_DIM, FEATURE_INDEX, FeatureScaler, FrameFeatureBuilder
from .evidence_accumulator import EvidenceAccumulator, EvidenceConfig, ReliabilityModel
from .fixed_temporal import TemporalVoter, VotingConfig
from .kalman_hsv import HSVCandidateConfig, KalmanBackgroundSubtractor, KalmanConfig
from .spatial_cnn import SpatialCNN, preprocess_crops

__all__ = [
    "AlarmPolicyConfig",
    "AlarmEvent",
    "AlarmPolicy",
    "DetectorConfig",
    "DecisionRecord",
    "CalibratedDetector",
    "VideoResult",
]


# --------------------------------------------------------------------------- #
# Alarm policy
# --------------------------------------------------------------------------- #
@dataclass
class AlarmPolicyConfig:
    """Configuration of the alarm decision layer.

    The ``fixed`` mode reproduces the original rule (a confidence threshold
    plus a fixed persistence duration). The ``adaptive`` mode modulates both
    the threshold and the required persistence using the model's uncertainty
    and the consistency of the motion evidence.
    """

    mode: str = "fixed"  # "fixed" | "adaptive"
    confidence_threshold: float = 0.70
    persistence_seconds: float = 1.50
    decision_window_seconds: float = 3.00
    refractory_seconds: float = 2.0
    hysteresis_ratio: float = 0.6
    # adaptive behaviour
    uncertainty_penalty: float = 0.6
    motion_bonus: float = 0.15
    min_threshold: float = 0.35
    max_threshold: float = 0.90
    max_persistence_seconds: float = 3.0
    # energy
    frame_skip: bool = False
    frame_skip_interval: int = 2
    frame_skip_confidence: float = 0.20

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence_threshold": self.confidence_threshold,
            "persistence_seconds": self.persistence_seconds,
            "decision_window_seconds": self.decision_window_seconds,
            "refractory_seconds": self.refractory_seconds,
            "hysteresis_ratio": self.hysteresis_ratio,
            "uncertainty_penalty": self.uncertainty_penalty,
            "motion_bonus": self.motion_bonus,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "max_persistence_seconds": self.max_persistence_seconds,
            "frame_skip": self.frame_skip,
            "frame_skip_interval": self.frame_skip_interval,
            "frame_skip_confidence": self.frame_skip_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AlarmPolicyConfig":
        cfg = cls()
        if not data:
            return cfg
        for key in (
            "confidence_threshold",
            "persistence_seconds",
            "decision_window_seconds",
            "refractory_seconds",
            "hysteresis_ratio",
            "uncertainty_penalty",
            "motion_bonus",
            "min_threshold",
            "max_threshold",
            "max_persistence_seconds",
            "frame_skip_confidence",
        ):
            if key in data:
                setattr(cfg, key, float(data[key]))
        for key in ("frame_skip",):
            if key in data:
                setattr(cfg, key, bool(data[key]))
        for key in ("mode",):
            if key in data:
                setattr(cfg, key, str(data[key]))
        if "frame_skip_interval" in data:
            cfg.frame_skip_interval = int(data["frame_skip_interval"])
        return cfg


@dataclass
class AlarmEvent:
    """One contiguous alarm episode."""

    video_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    peak_probability: float = 0.0
    peak_evidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "peak_probability": self.peak_probability,
            "peak_evidence": self.peak_evidence,
        }


class AlarmPolicy:
    """Hysteretic alarm state machine with an optional adaptive mode."""

    def __init__(self, config: AlarmPolicyConfig | None = None, fps: float = 25.0) -> None:
        self.config = config or AlarmPolicyConfig()
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.events: list[AlarmEvent] = []
        self.reset()

    def reset(self) -> None:
        """Reset the state machine and drop recorded events."""
        self.state: str = "idle"
        self.streak: int = 0
        self._open_event: AlarmEvent | None = None
        self._refractory_until: float = -1.0
        self.events = []

    # ------------------------------------------------------------------ core
    def effective_threshold(self, uncertainty: float, motion_reliability: float) -> float:
        """Threshold used for the current frame."""
        cfg = self.config
        if cfg.mode != "adaptive":
            return cfg.confidence_threshold
        threshold = (
            cfg.confidence_threshold
            + cfg.uncertainty_penalty * float(uncertainty)
            - cfg.motion_bonus * float(motion_reliability)
        )
        return float(np.clip(threshold, cfg.min_threshold, cfg.max_threshold))

    def required_persistence_frames(self, uncertainty: float) -> int:
        """Number of consecutive frames required to raise an alarm."""
        cfg = self.config
        seconds = cfg.persistence_seconds
        if cfg.mode == "adaptive":
            seconds = seconds * (1.0 + cfg.uncertainty_penalty * float(uncertainty))
        seconds = min(seconds, cfg.max_persistence_seconds)
        return max(1, int(round(seconds * self.fps)))

    def step(
        self,
        probability: float,
        *,
        frame_index: int,
        timestamp: float,
        uncertainty: float = 0.0,
        motion_reliability: float = 0.0,
        evidence: float = 0.0,
        video_id: str = "",
    ) -> dict[str, Any]:
        """Update the policy with one frame and return the decision record."""
        cfg = self.config
        threshold = self.effective_threshold(uncertainty, motion_reliability)
        required = self.required_persistence_frames(uncertainty)
        cancel_threshold = threshold * cfg.hysteresis_ratio

        decision = "continue_observing"
        fired = False

        if self.state == "alarm":
            if probability < cancel_threshold:
                if self._open_event is not None:
                    self._open_event.end_frame = frame_index
                    self._open_event.end_time = timestamp
                    self.events.append(self._open_event)
                    self._open_event = None
                self.state = "idle"
                self.streak = 0
                self._refractory_until = timestamp + cfg.refractory_seconds
                decision = "alarm_cleared"
            else:
                decision = "alarm_active"
        elif timestamp < self._refractory_until:
            decision = "refractory"
        else:
            if probability >= threshold:
                self.streak += 1
                self.state = "candidate"
                if self.streak >= required:
                    self.state = "alarm"
                    self.streak = 0
                    self._open_event = AlarmEvent(
                        video_id=video_id,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        start_time=timestamp,
                        end_time=timestamp,
                        peak_probability=float(probability),
                        peak_evidence=float(evidence),
                    )
                    decision = "alarm_raised"
                    fired = True
            else:
                self.streak = 0
                self.state = "idle"
                decision = "below_threshold"

        if self._open_event is not None:
            self._open_event.peak_probability = max(self._open_event.peak_probability, float(probability))
            self._open_event.peak_evidence = max(self._open_event.peak_evidence, float(evidence))

        return {
            "decision": decision,
            "state": self.state,
            "alarm": self.state == "alarm",
            "threshold": threshold,
            "required_persistence_frames": required,
            "fired": fired,
        }

    def close(self, frame_index: int, timestamp: float) -> None:
        """Close a still-open alarm event at the end of a video."""
        if self._open_event is not None:
            self._open_event.end_frame = frame_index
            self._open_event.end_time = timestamp
            self.events.append(self._open_event)
            self._open_event = None


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Full configuration of the detector."""

    temporal_kind: str = "tcn"  # "gru" | "tcn" | "evidence" | "voting" | "none"
    sequence_length: int = 16
    fps: float = 25.0
    kalman: dict[str, Any] = field(default_factory=dict)
    hsv: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    voting: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    spatial_input_size: tuple[int, int] = (50, 50)
    # Ablation switches.
    use_kalman_foreground: bool = True
    """When False the candidates are not gated by the Kalman motion mask."""

    use_hsv_filter: bool = True
    """When False the low-saturation constraint of the candidate mask is dropped."""

    zero_features: list[str] = field(default_factory=list)
    """Feature columns that are forced to zero before the temporal model.

    Used by the "TCN without handcrafted motion features" and "TCN without
    image-quality features" ablations.
    """

    def to_dict(self) -> dict:
        return {
            "temporal_kind": self.temporal_kind,
            "sequence_length": self.sequence_length,
            "fps": self.fps,
            "kalman": self.kalman,
            "hsv": self.hsv,
            "policy": self.policy,
            "voting": self.voting,
            "evidence": self.evidence,
            "spatial_input_size": list(self.spatial_input_size),
            "use_kalman_foreground": self.use_kalman_foreground,
            "use_hsv_filter": self.use_hsv_filter,
            "zero_features": list(self.zero_features),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "DetectorConfig":
        data = data or {}
        cfg = cls()
        if "temporal_kind" in data:
            cfg.temporal_kind = str(data["temporal_kind"])
        for key in ("sequence_length",):
            if key in data:
                setattr(cfg, key, int(data[key]))
        if "fps" in data:
            cfg.fps = float(data["fps"])
        for key in ("kalman", "hsv", "policy", "voting", "evidence"):
            if key in data:
                setattr(cfg, key, dict(data[key] or {}))
        if "spatial_input_size" in data:
            cfg.spatial_input_size = tuple(int(v) for v in data["spatial_input_size"])  # type: ignore[assignment]
        for key in ("use_kalman_foreground", "use_hsv_filter"):
            if key in data:
                setattr(cfg, key, bool(data[key]))
        if "zero_features" in data:
            cfg.zero_features = [str(v) for v in (data["zero_features"] or [])]
        return cfg


@dataclass
class DecisionRecord:
    """One row of the interpretable decision log."""

    video_id: str
    frame_index: int
    timestamp: float
    frame_probability: float
    calibrated_probability: float
    temporal_probability: float
    temporal_evidence: float
    uncertainty: float
    motion_reliability: float
    alarm_state: str
    decision: str
    threshold: float
    n_candidates: int = 0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "frame_probability": self.frame_probability,
            "calibrated_probability": self.calibrated_probability,
            "temporal_probability": self.temporal_probability,
            "temporal_evidence": self.temporal_evidence,
            "uncertainty": self.uncertainty,
            "motion_reliability": self.motion_reliability,
            "alarm_state": self.alarm_state,
            "decision": self.decision,
            "threshold": self.threshold,
            "n_candidates": self.n_candidates,
            "skipped": self.skipped,
        }


@dataclass
class VideoResult:
    """Everything produced by running the detector on one video."""

    video_id: str
    fps: float
    num_frames: int
    decisions: list[DecisionRecord]
    events: list[AlarmEvent]
    frame_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    temporal_probabilities: np.ndarray
    frame_indices: np.ndarray
    latencies: np.ndarray
    skipped_frames: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "n_events": len(self.events),
            "n_decisions": len(self.decisions),
            "skipped_frames": self.skipped_frames,
            "mean_latency_ms": float(np.mean(self.latencies)) if self.latencies.size else 0.0,
        }


class CalibratedDetector:
    """End-to-end calibrated temporal smoke detector."""

    def __init__(
        self,
        *,
        spatial_model: SpatialCNN | None = None,
        temporal_model: torch.nn.Module | None = None,
        calibrator=None,
        policy_config: AlarmPolicyConfig | None = None,
        voter: TemporalVoter | None = None,
        evidence: EvidenceAccumulator | None = None,
        feature_scaler: FeatureScaler | None = None,
        config: DetectorConfig | None = None,
        device: str | torch.device = "cpu",
        video_id: str = "",
    ) -> None:
        self.config = config or DetectorConfig()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.calibrator = calibrator
        self.feature_scaler = feature_scaler
        self.device = torch.device(device)
        self.video_id = video_id

        self.kalman_cfg = KalmanConfig(**(self.config.kalman or {}))
        self.hsv_cfg = HSVCandidateConfig(**(self.config.hsv or {}))
        self.policy = AlarmPolicy(policy_config or AlarmPolicyConfig.from_dict(self.config.policy), fps=self.config.fps)
        self.voter = voter or TemporalVoter(VotingConfig.from_dict(self.config.voting), fps=self.config.fps)
        self.evidence = evidence or EvidenceAccumulator(
            EvidenceConfig.from_dict(self.config.evidence),
            ReliabilityModel.from_state_dict((self.config.evidence or {}).get("reliability_model")),
        )

        self.kalman = KalmanBackgroundSubtractor(self.kalman_cfg)
        self.feature_builder = FrameFeatureBuilder(max_candidates=self.hsv_cfg.max_candidates)

        self._history: deque[np.ndarray] = deque(maxlen=self.config.sequence_length)
        self._motion_history: deque[float] = deque(maxlen=self.config.sequence_length)
        self._last_candidates: list = []
        self._last_feature: np.ndarray | None = None
        self._last_frame_logit: float = 0.0
        self._last_latency_ms: float = 0.0
        self._skip_countdown: int = 0
        self._skipped_total: int = 0
        # Per-stage wall-clock accumulators, used by the edge benchmark to
        # report the cost of every stage of the pipeline separately.
        self.stage_times: dict[str, float] = {
            "decoding": 0.0,
            "kalman": 0.0,
            "hsv_candidates": 0.0,
            "spatial_cnn": 0.0,
            "features": 0.0,
            "temporal": 0.0,
            "alarm_policy": 0.0,
            "logging": 0.0,
        }

        if self.spatial_model is not None:
            self.spatial_model.to(self.device).eval()
        if self.temporal_model is not None:
            self.temporal_model.to(self.device).eval()

    # ------------------------------------------------------------------ state
    def reset(self, video_id: str | None = None) -> None:
        """Reset all per-video state."""
        if video_id is not None:
            self.video_id = video_id
        self.kalman.reset()
        self.feature_builder.reset()
        self.policy = AlarmPolicy(self.policy.config, fps=self.config.fps)
        self.voter.reset()
        self.evidence.reset()
        self._history.clear()
        self._motion_history.clear()
        self._last_candidates = []
        self._last_feature = None
        self._last_frame_logit = 0.0
        self._last_latency_ms = 0.0
        self._skip_countdown = 0
        self._skipped_total = 0
        for key in self.stage_times:
            self.stage_times[key] = 0.0

    # ------------------------------------------------------------- front end
    def _spatial_probabilities(self, candidates: Sequence) -> np.ndarray:
        """Run the spatial CNN on the candidate crops of a frame."""
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        if self.spatial_model is None:
            return np.zeros(len(candidates), dtype=np.float32)
        crops = np.stack([c.crop for c in candidates])
        tensor = preprocess_crops(crops, self.config.spatial_input_size).to(self.device)
        with torch.no_grad():
            logits = self.spatial_model(tensor).detach().cpu().numpy()
        return logits.astype(np.float64)

    def _motion_reliability(self) -> float:
        """Consistency of the motion signal over the recent window, in [0, 1]."""
        if len(self._motion_history) < 2:
            return 0.0
        values = np.asarray(self._motion_history, dtype=np.float64)
        mean = float(values.mean())
        if mean <= 1e-6:
            return 0.0
        cv = float(values.std() / mean)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    # ------------------------------------------------------------- inference
    def _temporal_inference(self, feature: np.ndarray) -> tuple[float, float, float, float]:
        """Return ``(temporal_prob, evidence, uncertainty, frame_prob)``.

        ``frame_prob`` is the calibrated probability of the *current frame* as
        produced by the spatial model, and is reported for comparison.
        """
        kind = self.config.temporal_kind

        # Calibrated frame probability (spatial model output).
        frame_logit = float(self._last_frame_logit)
        if self.calibrator is not None:
            frame_prob = float(self.calibrator.predict_proba(np.array([frame_logit]))[0])
        else:
            frame_prob = float(1.0 / (1.0 + np.exp(-np.clip(frame_logit, -60, 60))))

        if kind == "none":
            return frame_prob, 0.0, 0.0, frame_prob

        if kind == "voting":
            out = self.voter.step(frame_prob)
            return float(out["probability"]), 0.0, 0.0, frame_prob

        if kind == "evidence":
            from .evidence_accumulator import reliability_features

            motion_consistency = self._motion_reliability()
            rel_feats = reliability_features(feature, uncertainty=0.0, motion_consistency=motion_consistency)
            out = self.evidence.step(frame_prob, reliability_features=rel_feats)
            evidence_value = float(out["evidence"])
            alarm_state = str(out["state"])
            temporal_prob = float(1.0 / (1.0 + np.exp(-np.clip(evidence_value / 4.0, -30, 30))))
            return temporal_prob, evidence_value, float(out["reliability"]), frame_prob

        # Neural temporal models.
        if self.temporal_model is None:
            return frame_prob, 0.0, 0.0, frame_prob

        self._history.append(feature)
        if len(self._history) < self.config.sequence_length:
            pad = np.repeat(self._history[0][None, :], self.config.sequence_length - len(self._history), axis=0)
            sequence = np.concatenate([pad, np.asarray(self._history)], axis=0)
        else:
            sequence = np.asarray(self._history)

        if self.feature_scaler is not None and self.feature_scaler.fitted:
            sequence = self.feature_scaler.transform(sequence)

        tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.temporal_model(tensor)
        now_logit = float(out["now"].reshape(-1)[0].item())
        log_var = float(out["log_variance"].reshape(-1)[0].item())
        uncertainty = float(np.exp(0.5 * np.clip(log_var, -20.0, 6.0)))

        if self.calibrator is not None:
            temporal_prob = float(self.calibrator.predict_proba(np.array([now_logit]))[0])
        else:
            temporal_prob = float(1.0 / (1.0 + np.exp(-np.clip(now_logit, -60, 60))))

        return temporal_prob, 0.0, uncertainty, frame_prob

    # ---------------------------------------------------------------- public
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp: float | None = None,
    ) -> DecisionRecord:
        """Process a single frame and return the decision record."""
        start = time.perf_counter()
        ts = float(frame_index / self.config.fps) if timestamp is None else float(timestamp)

        skipped = False
        if self.policy.config.frame_skip and self._skip_countdown > 0:
            skipped = True
            self._skip_countdown -= 1
            self._skipped_total += 1

        if skipped and self._last_feature is not None:
            feature = self._last_feature.copy()
            n_candidates = len(self._last_candidates)
        else:
            t0 = time.perf_counter()
            fg_mask, residual = self.kalman.apply(frame_bgr)
            t1 = time.perf_counter()
            candidates = self._models_extract(frame_bgr, fg_mask, residual, frame_index)
            t2 = time.perf_counter()
            logits = self._spatial_probabilities(candidates)
            t3 = time.perf_counter()
            self._last_frame_logit = float(logits.max()) if logits.size else 0.0
            self._last_candidates = candidates
            fg_ratio = float((fg_mask > 0).mean())
            brightness, blur = self._frame_quality(frame_bgr)
            feature = self.feature_builder.build(
                frame_index=frame_index,
                candidates=candidates,
                cnn_probs=list(logits),
                brightness=brightness,
                blur_score=blur,
                foreground_ratio=fg_ratio,
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
            )
            t4 = time.perf_counter()
            self._last_feature = feature
            n_candidates = len(candidates)
            self._motion_history.append(float(feature[FEATURE_INDEX["motion_magnitude"]]))
            self.stage_times["kalman"] += t1 - t0
            self.stage_times["hsv_candidates"] += t2 - t1
            self.stage_times["spatial_cnn"] += t3 - t2
            self.stage_times["features"] += t4 - t3

        if self.config.zero_features:
            feature = feature.copy()
            for name in self.config.zero_features:
                index = FEATURE_INDEX.get(name)
                if index is not None:
                    feature[index] = 0.0

        t0 = time.perf_counter()
        temporal_prob, evidence, uncertainty, frame_prob = self._temporal_inference(feature)
        motion_reliability = self._motion_reliability()
        t1 = time.perf_counter()

        policy_out = self.policy.step(
            temporal_prob,
            frame_index=frame_index,
            timestamp=ts,
            uncertainty=uncertainty,
            motion_reliability=motion_reliability,
            evidence=evidence,
            video_id=self.video_id,
        )
        t2 = time.perf_counter()
        self.stage_times["temporal"] += t1 - t0
        self.stage_times["alarm_policy"] += t2 - t1

        if (
            self.policy.config.frame_skip
            and not skipped
            and temporal_prob < self.policy.config.frame_skip_confidence
            and n_candidates == 0
        ):
            self._skip_countdown = max(0, self.policy.config.frame_skip_interval - 1)

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = latency_ms
        return DecisionRecord(
            video_id=self.video_id,
            frame_index=frame_index,
            timestamp=ts,
            frame_probability=frame_prob,
            calibrated_probability=frame_prob,
            temporal_probability=temporal_prob,
            temporal_evidence=evidence,
            uncertainty=uncertainty,
            motion_reliability=motion_reliability,
            alarm_state=str(policy_out["state"]),
            decision=str(policy_out["decision"]),
            threshold=float(policy_out["threshold"]),
            n_candidates=n_candidates,
            skipped=skipped,
        )

    def _models_extract(self, frame_bgr, fg_mask, residual, frame_index):
        """Extract the candidates of a frame, honouring the ablation switches."""
        from .kalman_hsv import extract_candidates

        if not self.config.use_kalman_foreground:
            # Ablation "no Kalman motion model": the candidates are not gated
            # by the motion mask.
            fg_mask = None
        elif not self.config.use_hsv_filter:
            # Ablation "no HSV filtering": the candidate mask is the motion
            # mask itself, without the low-saturation constraint.
            fg_mask = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)

        return extract_candidates(
            frame_bgr,
            frame_index,
            foreground_mask=fg_mask,
            residual=residual,
            config=self.hsv_cfg,
        )

    @staticmethod
    def _frame_quality(frame_bgr: np.ndarray) -> tuple[float, float]:
        from .kalman_hsv import image_quality

        return image_quality(frame_bgr)

    def process_video(self, path: str, video_id: str | None = None, max_frames: int | None = None) -> VideoResult:
        """Run the detector over an entire video file."""
        import cv2

        vid = video_id or self.video_id or "video"
        self.reset(vid)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or self.config.fps
        if not fps or fps <= 0:
            fps = self.config.fps
        self.config.fps = fps
        self.policy = AlarmPolicy(self.policy.config, fps=fps)
        self.voter = TemporalVoter(self.voter.config, fps=fps)

        decisions: list[DecisionRecord] = []
        latencies: list[float] = []
        index = 0
        while True:
            t0 = time.perf_counter()
            ok, frame = capture.read()
            t1 = time.perf_counter()
            if not ok:
                break
            record = self.process_frame(frame, index, timestamp=index / fps)
            t2 = time.perf_counter()
            decisions.append(record)
            latencies.append(self._last_latency_ms)
            self.stage_times["decoding"] += t1 - t0
            self.stage_times["logging"] += t2 - t1
            index += 1
            if max_frames is not None and index >= max_frames:
                break
        capture.release()
        self.policy.close(index, index / fps)

        frame_probs = np.array([d.frame_probability for d in decisions], dtype=np.float64)
        cal_probs = np.array([d.calibrated_probability for d in decisions], dtype=np.float64)
        temp_probs = np.array([d.temporal_probability for d in decisions], dtype=np.float64)
        return VideoResult(
            video_id=vid,
            fps=fps,
            num_frames=index,
            decisions=decisions,
            events=list(self.policy.events),
            frame_probabilities=frame_probs,
            calibrated_probabilities=cal_probs,
            temporal_probabilities=temp_probs,
            frame_indices=np.arange(index),
            latencies=np.asarray(latencies, dtype=np.float64),
            skipped_frames=self._skipped_total,
        )
