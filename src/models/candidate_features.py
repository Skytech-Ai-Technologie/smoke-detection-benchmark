"""Handcrafted candidate features and the per-frame feature vector.

The temporal models consume a fixed-length vector per frame (16 features by
default). The vector summarises

* the spatial model output (CNN smoke probability),
* the geometry of the strongest candidate (area, size, aspect ratio),
* temporal evidence (area change, candidate persistence),
* appearance (mean HSV of the candidate),
* image quality (brightness, blur) and scene activity (foreground ratio).

Keeping the CNN probability as *one* feature (rather than letting the
temporal model see pixels) is what makes the cascade cheap enough to run on
edge hardware, and it keeps the temporal model comparable against the fixed
voting baselines that also only see frame probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_DIM",
    "FEATURE_INDEX",
    "CandidateTracker",
    "FrameFeatureBuilder",
    "FeatureScaler",
    "build_frame_features",
]

FEATURE_NAMES: list[str] = [
    "cnn_prob_max",       # strongest spatial smoke probability in the frame
    "cnn_prob_mean",      # mean spatial probability over all candidates
    "area_ratio",         # candidate area / frame area
    "area_change",        # log area ratio w.r.t. the previous frame of the track
    "motion_magnitude",   # mean colour residual inside the candidate
    "bbox_width",         # bounding-box width / frame width
    "bbox_height",        # bounding-box height / frame height
    "aspect_ratio",       # width / height of the bounding box
    "hsv_h",              # mean hue of the candidate (0-179, OpenCV scale)
    "hsv_s",              # mean saturation of the candidate (0-255)
    "hsv_v",              # mean value of the candidate (0-255)
    "persistence",        # number of consecutive frames the track exists
    "brightness",         # mean frame intensity (0-255)
    "blur_score",         # variance of the Laplacian
    "n_candidates",       # number of candidates / max_candidates
    "foreground_ratio",   # fraction of pixels flagged by the Kalman model
]

FEATURE_DIM: int = len(FEATURE_NAMES)
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class TrackedCandidate:
    """State of one candidate track across frames."""

    track_id: int
    bbox: tuple[int, int, int, int]
    area: int
    age: int = 1
    missed: int = 0
    last_area: int = 0

    def iou_with(self, bbox: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = self.bbox
        bx0, by0, bx1, by1 = bbox
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


class CandidateTracker:
    """A minimal IoU-based tracker used to build persistence / area-change features.

    It is intentionally simple: the purpose is to provide *temporal context
    features* to the temporal model, not to be a state-of-the-art tracker.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[TrackedCandidate] = []
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update(self, candidates: Sequence) -> dict[int, TrackedCandidate]:
        """Match ``candidates`` (objects with ``bbox_xyxy`` / ``area``) to tracks.

        Returns:
            A mapping ``candidate_id -> TrackedCandidate``.
        """
        matches: dict[int, TrackedCandidate] = {}
        used_tracks: set[int] = set()

        for cand in candidates:
            best_track = None
            best_iou = self.iou_threshold
            for track in self._tracks:
                if track.track_id in used_tracks:
                    continue
                iou = track.iou_with(cand.bbox_xyxy)
                if iou >= best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is None:
                best_track = TrackedCandidate(
                    track_id=self._next_id,
                    bbox=cand.bbox_xyxy,
                    area=cand.area,
                    age=1,
                    missed=0,
                    last_area=cand.area,
                )
                self._next_id += 1
                self._tracks.append(best_track)
            else:
                used_tracks.add(best_track.track_id)
                best_track.last_area = best_track.area
                best_track.bbox = cand.bbox_xyxy
                best_track.area = cand.area
                best_track.age += 1
                best_track.missed = 0
            matches[cand.candidate_id] = best_track

        # Retire tracks that were not seen for too long.
        survivors: list[TrackedCandidate] = []
        for track in self._tracks:
            if track.track_id in used_tracks:
                survivors.append(track)
                continue
            track.missed += 1
            if track.missed <= self.max_missed:
                survivors.append(track)
        self._tracks = survivors
        return matches


class FrameFeatureBuilder:
    """Turn per-frame candidate metadata into the temporal model's feature matrix.

    Args:
        max_candidates: Reference count used to normalise the candidate count.
        max_persistence: Reference persistence used to normalise the
            persistence feature.
    """

    def __init__(self, max_candidates: int = 8, max_persistence: int = 64) -> None:
        self.max_candidates = max_candidates
        self.max_persistence = max_persistence
        self.tracker = CandidateTracker()

    def reset(self) -> None:
        self.tracker.reset()

    def build(
        self,
        *,
        frame_index: int,
        candidates: Sequence,
        cnn_probs: Sequence[float] | None,
        brightness: float,
        blur_score: float,
        foreground_ratio: float,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Return the ``(FEATURE_DIM,)`` feature vector for a single frame."""
        frame_area = max(1, frame_width * frame_height)
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)

        if cnn_probs is None or len(cnn_probs) != len(candidates):
            probs = np.zeros(len(candidates), dtype=np.float32)
        else:
            probs = np.asarray(cnn_probs, dtype=np.float32)

        feat[FEATURE_INDEX["n_candidates"]] = min(
            len(candidates) / float(self.max_candidates), 1.0
        )
        feat[FEATURE_INDEX["brightness"]] = brightness / 255.0
        feat[FEATURE_INDEX["blur_score"]] = float(np.log1p(max(0.0, blur_score)) / 10.0)
        feat[FEATURE_INDEX["foreground_ratio"]] = float(np.clip(foreground_ratio, 0.0, 1.0))

        if not candidates:
            return feat

        matches = self.tracker.update(candidates)

        # "Best" candidate: highest spatial probability, ties broken by area.
        best_idx = int(np.argmax(probs)) if probs.size else 0
        if probs.size and probs[best_idx] <= 0.0:
            best_idx = int(
                max(range(len(candidates)), key=lambda i: candidates[i].area)
            )

        feat[FEATURE_INDEX["cnn_prob_max"]] = float(probs[best_idx])
        feat[FEATURE_INDEX["cnn_prob_mean"]] = float(np.mean(probs)) if probs.size else 0.0

        best = candidates[best_idx]
        track = matches.get(best.candidate_id)

        feat[FEATURE_INDEX["area_ratio"]] = float(np.clip(best.area / frame_area, 0.0, 1.0))
        feat[FEATURE_INDEX["motion_magnitude"]] = float(best.motion_magnitude / 255.0)
        feat[FEATURE_INDEX["bbox_width"]] = float(best.width / max(1, frame_width))
        feat[FEATURE_INDEX["bbox_height"]] = float(best.height / max(1, frame_height))
        feat[FEATURE_INDEX["aspect_ratio"]] = float(np.clip(best.aspect_ratio, 0.0, 20.0)) / 20.0
        feat[FEATURE_INDEX["hsv_h"]] = float(best.mean_hsv[0]) / 180.0
        feat[FEATURE_INDEX["hsv_s"]] = float(best.mean_hsv[1]) / 255.0
        feat[FEATURE_INDEX["hsv_v"]] = float(best.mean_hsv[2]) / 255.0

        if track is not None:
            feat[FEATURE_INDEX["persistence"]] = min(track.age / float(self.max_persistence), 1.0)
            prev_area = max(1, track.last_area)
            change = np.log(max(1, best.area) / float(prev_area))
            feat[FEATURE_INDEX["area_change"]] = float(np.clip(change, -3.0, 3.0)) / 3.0

        return feat


def build_frame_features(
    per_frame_records: Sequence[dict],
    cnn_probs: Sequence[Sequence[float]] | None = None,
    *,
    max_candidates: int = 8,
) -> np.ndarray:
    """Vectorised convenience wrapper around :class:`FrameFeatureBuilder`.

    Args:
        per_frame_records: One dict per frame with keys ``candidates``,
            ``brightness``, ``blur_score``, ``foreground_ratio``,
            ``frame_width`` and ``frame_height``.
        cnn_probs: Optional per-frame list of per-candidate CNN probabilities.

    Returns:
        A ``(T, FEATURE_DIM)`` float32 matrix.
    """
    builder = FrameFeatureBuilder(max_candidates=max_candidates)
    out = np.zeros((len(per_frame_records), FEATURE_DIM), dtype=np.float32)
    for i, rec in enumerate(per_frame_records):
        probs = cnn_probs[i] if cnn_probs is not None else None
        out[i] = builder.build(
            frame_index=i,
            candidates=rec.get("candidates", []),
            cnn_probs=probs,
            brightness=rec.get("brightness", 0.0),
            blur_score=rec.get("blur_score", 0.0),
            foreground_ratio=rec.get("foreground_ratio", 0.0),
            frame_width=rec.get("frame_width", 1),
            frame_height=rec.get("frame_height", 1),
        )
    return out


@dataclass
class FeatureScaler:
    """Standardise feature columns using statistics fitted on training videos only."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(FEATURE_DIM, dtype=np.float32))
    fitted: bool = False

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (N, {FEATURE_DIM}) features, got {features.shape}")
        self.mean = features.mean(axis=0).astype(np.float32)
        std = features.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.std = std
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        return ((features - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted": self.fitted}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "FeatureScaler":
        scaler = cls()
        if state.get("fitted"):
            scaler.mean = np.asarray(state["mean"], dtype=np.float32)
            scaler.std = np.asarray(state["std"], dtype=np.float32)
            scaler.fitted = True
        return scaler
