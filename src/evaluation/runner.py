"""Shared evaluation runner.

The runner executes a detector over a list of videos and returns, per video,

* the per-frame decision log,
* the per-frame score used for the frame-level metrics,
* the frame labels derived from the smoke onset,
* the alarm events.

The smoke onset is read from the cached feature archive so that the
evaluation uses exactly the same onset that the training pipeline resolved
(annotation when available, documented weak-onset heuristic otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.extract_frames import load_features
from ..models.builder import build_detector
from ..models.candidate_features import FEATURE_INDEX

__all__ = ["VideoSpec", "RunResult", "run_detector", "load_specs_from_manifest"]


@dataclass
class VideoSpec:
    """Everything needed to evaluate one video."""

    video_id: str
    path: str
    label: str
    onset_frame: int | None = None
    fps: float = 25.0
    dataset: str = ""

    @property
    def is_positive(self) -> bool:
        return self.label == "smoke"


@dataclass
class RunResult:
    """The result of running a detector over one video."""

    spec: VideoSpec
    decisions: list[Any] = field(default_factory=list)
    frame_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spatial_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frame_labels: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list[Any] = field(default_factory=list)
    fps: float = 25.0
    num_frames: int = 0
    skipped_frames: int = 0
    latencies_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def alarm_mask(self) -> np.ndarray:
        """Boolean mask of the frames during which the alarm was active."""
        return np.array([d.alarm_state == "alarm" for d in self.decisions], dtype=bool)

    def summary(self) -> dict[str, Any]:
        return {
            "video_id": self.spec.video_id,
            "label": self.spec.label,
            "dataset": self.spec.dataset,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "onset_frame": self.spec.onset_frame,
            "n_events": len(self.events),
            "skipped_frames": self.skipped_frames,
        }


def load_specs_from_manifest(
    manifest_path: str | Path,
    feature_dir: str | Path | None = None,
    *,
    role: str | None = None,
    split: str | None = None,
    splits_file: str | Path | None = None,
) -> list[VideoSpec]:
    """Build the list of video specifications to evaluate.

    Args:
        manifest_path: The dataset manifest (one row per video).
        feature_dir: Directory with the cached feature archives; used to
            resolve the smoke onset.
        role: Restrict to ``train`` / ``validation`` / ``test``.
        split: Split id (requires ``splits_file``).
        splits_file: The split registry.
    """
    from ..data.video_split import read_manifest

    rows = read_manifest(manifest_path)
    allowed: set[str] | None = None
    if split is not None and splits_file is not None:
        allowed = {
            row["video_id"]
            for row in read_manifest(splits_file)
            if row.get("split_id") == split and (role is None or row.get("role") == role)
        }

    specs: list[VideoSpec] = []
    for row in rows:
        video_id = row["video_id"]
        if allowed is not None and video_id not in allowed:
            continue
        if role is not None and allowed is None and row.get("split") != role:
            continue

        onset: int | None = None
        fps = float(row.get("fps") or 25.0)
        if feature_dir is not None:
            archive = Path(feature_dir) / f"{video_id}.npz"
            if archive.exists():
                data = load_features(archive)
                onset_value = int(data["onset_frame"])
                onset = None if onset_value < 0 else onset_value
                fps = float(data["fps"]) or fps
        if onset is None:
            raw = str(row.get("onset_frame", "")).strip()
            if raw not in ("", "None", "nan"):
                try:
                    onset = int(float(raw))
                except ValueError:
                    onset = None

        specs.append(
            VideoSpec(
                video_id=video_id,
                path=row["path"],
                label=row.get("label", "unknown"),
                onset_frame=onset,
                fps=fps,
                dataset=row.get("dataset", ""),
            )
        )
    return specs


def run_detector(
    specs: Sequence[VideoSpec],
    *,
    spatial_checkpoint: str | Path | None = None,
    temporal_checkpoint: str | Path | None = None,
    device: str = "cpu",
    policy_overrides: dict[str, Any] | None = None,
    score_source: str = "temporal",
    detector_config=None,
    max_frames: int | None = None,
    variant: str | None = None,
) -> list[RunResult]:
    """Run a detector over a list of videos.

    Args:
        specs: The videos to process.
        spatial_checkpoint: Spatial CNN checkpoint.
        temporal_checkpoint: Temporal checkpoint.
        device: Torch device.
        policy_overrides: Overrides for the alarm policy.
        score_source: ``temporal`` (the detector output) or ``spatial`` (the
            calibrated frame probability of the spatial model).
        detector_config: Optional :class:`DetectorConfig`.
        max_frames: Optional frame limit (used by the latency benchmark).
    """
    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides=policy_overrides,
        detector_config=detector_config,
        variant=variant,
    )
    detector = bundle.detector

    results: list[RunResult] = []
    for spec in specs:
        detector.reset(spec.video_id)
        result = detector.process_video(spec.path, video_id=spec.video_id, max_frames=max_frames)
        mask = np.array([d.alarm_state == "alarm" for d in result.decisions], dtype=bool)

        temporal = np.asarray(result.temporal_probabilities, dtype=np.float64)
        spatial = np.asarray(result.calibrated_probabilities, dtype=np.float64)
        scores = temporal if score_source == "temporal" else spatial

        n = len(result.decisions)
        labels = np.zeros(n, dtype=np.int64)
        if spec.is_positive and spec.onset_frame is not None:
            labels[spec.onset_frame :] = 1

        results.append(
            RunResult(
                spec=spec,
                decisions=result.decisions,
                frame_scores=scores,
                spatial_scores=spatial,
                frame_labels=labels,
                events=result.events,
                fps=result.fps,
                num_frames=result.num_frames,
                skipped_frames=result.skipped_frames,
                latencies_ms=result.latencies,
            )
        )
    return results


def aggregate_frame_scores(results: Sequence[RunResult]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the frame scores and labels of several runs."""
    if not results:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    scores = np.concatenate([r.frame_scores for r in results])
    labels = np.concatenate([r.frame_labels for r in results])
    return scores, labels
