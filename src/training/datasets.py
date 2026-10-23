"""Training data assembly.

The cached per-video feature archives are turned into

* a **crop dataset** for the spatial CNN (crops + three-valued labels), and
* a **sequence dataset** for the temporal models (windows of per-frame
  features + a current/horizon label pair).

Nothing in this module ever touches the test split: the caller decides which
role a video plays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.extract_frames import LABEL_BACKGROUND, LABEL_IGNORE, LABEL_SMOKE, load_features
from ..models.candidate_features import (
    FEATURE_DIM,
    FEATURE_INDEX,
    FeatureScaler,
    FrameFeatureBuilder,
)
from ..models.kalman_hsv import Candidate

__all__ = [
    "VideoFeatures",
    "load_video_features",
    "rebuild_candidates",
    "frame_features_for_video",
    "score_candidates",
    "CropDataset",
    "FrameCropDataset",
    "mil_collate",
    "collect_frame_groups",
    "TemporalSequenceDataset",
    "build_temporal_dataset",
    "frame_labels_for_video",
]


@dataclass
class VideoFeatures:
    """The cached arrays of one video."""

    video_id: str
    label: str
    fps: float
    num_frames: int
    width: int
    height: int
    onset_frame: int | None
    onset_source: str
    candidates: list[Candidate]
    crops: np.ndarray
    cand_labels: np.ndarray
    brightness: np.ndarray
    blur: np.ndarray
    fg_ratio: np.ndarray
    frame_motion: np.ndarray
    max_area_ratio: np.ndarray
    cand_frame_index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    frame_label: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


def load_video_features(path: str | Path) -> VideoFeatures:
    """Load a cached feature archive into a :class:`VideoFeatures` object."""
    data = load_features(path)
    onset = int(data["onset_frame"])
    candidates = rebuild_candidates(data)
    return VideoFeatures(
        video_id=str(data["video_id"]),
        label=str(data["label"]),
        fps=float(data["fps"]),
        num_frames=int(data["num_frames"]),
        width=int(data["width"]),
        height=int(data["height"]),
        onset_frame=None if onset < 0 else onset,
        onset_source=str(data.get("onset_source", "unknown")),
        candidates=candidates,
        crops=data.get("cand_crop", np.zeros((0, 50, 50, 3), np.uint8)),
        cand_labels=data.get("cand_label", np.zeros(0, np.int8)),
        cand_frame_index=data.get("cand_frame_index", np.zeros(0, np.int32)),
        brightness=data.get("frame_brightness", np.zeros(0, np.float32)),
        blur=data.get("frame_blur", np.zeros(0, np.float32)),
        fg_ratio=data.get("frame_fg_ratio", np.zeros(0, np.float32)),
        frame_motion=data.get("frame_motion", np.zeros(0, np.float32)),
        max_area_ratio=data.get("frame_max_area_ratio", np.zeros(0, np.float32)),
        frame_label=data.get("frame_label", np.zeros(0, np.int8)),
    )


def rebuild_candidates(data: dict[str, Any]) -> list[Candidate]:
    """Reconstruct :class:`Candidate` objects from the cached arrays."""
    frame_index = data.get("cand_frame_index", np.zeros(0, np.int32))
    bboxes = data.get("cand_bbox", np.zeros((0, 4), np.int32))
    areas = data.get("cand_area", np.zeros(0, np.int32))
    hsv = data.get("cand_hsv", np.zeros((0, 3), np.float32))
    motion = data.get("cand_motion", np.zeros(0, np.float32))

    out: list[Candidate] = []
    for i in range(len(frame_index)):
        bbox = tuple(int(v) for v in bboxes[i]) if len(bboxes) else (0, 0, 0, 0)
        out.append(
            Candidate(
                frame_index=int(frame_index[i]),
                candidate_id=i,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                area=int(areas[i]) if len(areas) else 0,
                mean_hsv=tuple(float(v) for v in hsv[i]) if len(hsv) else (0.0, 0.0, 0.0),
                motion_magnitude=float(motion[i]) if len(motion) else 0.0,
            )
        )
    return out


def frame_features_for_video(
    video: VideoFeatures,
    cnn_logits: Sequence[float] | None = None,
    *,
    max_candidates: int = 8,
) -> np.ndarray:
    """Build the ``(T, FEATURE_DIM)`` feature matrix of one video.

    Args:
        video: The cached video features.
        cnn_logits: Optional per-candidate spatial logits. When ``None`` the
            candidate-selection falls back to the largest candidate, which is
            what happens before the spatial model has been trained.
    """
    builder = FrameFeatureBuilder(max_candidates=max_candidates)
    n_frames = max(1, len(video.brightness))
    features = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)

    # Group candidates by frame.
    by_frame: dict[int, list[int]] = {}
    for idx, cand in enumerate(video.candidates):
        by_frame.setdefault(cand.frame_index, []).append(idx)

    for t in range(n_frames):
        indices = by_frame.get(t, [])
        candidates = [video.candidates[i] for i in indices]
        logits = [cnn_logits[i] for i in indices] if cnn_logits is not None else None
        features[t] = builder.build(
            frame_index=t,
            candidates=candidates,
            cnn_probs=logits,
            brightness=float(video.brightness[t]) if t < len(video.brightness) else 0.0,
            blur_score=float(video.blur[t]) if t < len(video.blur) else 0.0,
            foreground_ratio=float(video.fg_ratio[t]) if t < len(video.fg_ratio) else 0.0,
            frame_width=max(1, video.width),
            frame_height=max(1, video.height),
        )
    return features


def score_candidates(
    model: torch.nn.Module,
    video: VideoFeatures,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run the spatial model over all candidate crops of a video."""
    from ..models.spatial_cnn import preprocess_crops

    model = model.to(device).eval()
    if video.crops.size == 0:
        return np.zeros(0, dtype=np.float64)

    logits = np.zeros(video.crops.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, video.crops.shape[0], batch_size):
            batch = video.crops[start : start + batch_size]
            tensor = preprocess_crops(batch).to(device)
            out = model(tensor).detach().cpu().numpy().reshape(-1)
            logits[start : start + batch_size] = out
    return logits


def frame_labels_for_video(
    video: VideoFeatures,
    horizon: int = 8,
    guard_frames: int = 0,
) -> dict[str, np.ndarray]:
    """Per-frame labels for the temporal models.

    Returns:
        A dictionary with ``y_now`` (smoke in the current frame), ``y_horizon``
        (smoke within the next ``horizon`` frames), ``mask_now`` and
        ``mask_horizon`` (loss masks; ambiguous frames are excluded).
    """
    n = max(1, len(video.brightness))
    y_now = np.zeros(n, dtype=np.float32)
    y_horizon = np.zeros(n, dtype=np.float32)
    mask_now = np.ones(n, dtype=bool)
    mask_horizon = np.ones(n, dtype=bool)

    if video.label == "smoke" and video.onset_frame is not None:
        onset = int(video.onset_frame)
        y_now[onset:] = 1.0
        y_horizon[: max(0, onset + 1)] = (np.arange(max(0, onset + 1)) + horizon >= onset).astype(np.float32)
        # Frames immediately before the onset are ambiguous for the *current*
        # label (smoke may already be faintly visible) and are masked out.
        lo = max(0, onset - guard_frames)
        mask_now[lo:onset] = False
    elif video.label == "smoke":
        # No usable onset: the video cannot contribute a positive label.
        mask_now[:] = False
        mask_horizon[:] = False

    return {
        "y_now": y_now,
        "y_horizon": y_horizon,
        "mask_now": mask_now,
        "mask_horizon": mask_horizon,
    }


class CropDataset(Dataset):
    """Crops of the spatial classifier."""

    def __init__(
        self,
        crops: np.ndarray,
        labels: np.ndarray,
        *,
        input_size: tuple[int, int] = (50, 50),
        augment: bool = False,
        augment_config=None,
        seed: int = 0,
    ) -> None:
        self.crops = crops
        self.labels = labels.astype(np.float32)
        self.input_size = input_size
        self.augment = augment
        self.augment_config = augment_config
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        from ..data.augment import augment_crop
        from ..models.spatial_cnn import preprocess_crops

        crop = self.crops[index]
        if self.augment and self.augment_config is not None:
            crop = augment_crop(crop, self.rng, self.augment_config)
        tensor = preprocess_crops(crop[None, ...], self.input_size)[0]
        return tensor, torch.tensor(self.labels[index], dtype=torch.float32)


class FrameCropDataset(Dataset):
    """Weakly-supervised (MIL) dataset: one item per frame, many crops.

    FIRESENSE provides no region-level smoke annotation, so the spatial
    classifier cannot be trained with a per-crop label. Instead the
    *frame* label is used with a multiple-instance objective:

    * a **positive** frame must contain at least one smoke-like candidate
      (the loss is applied to the maximum candidate probability),
    * a **negative** frame must not contain any smoke-like candidate (the
      loss is applied to the mean candidate probability).

    This is the standard weakly-supervised formulation and it matches the
    decision rule of the detector, which fires when *any* candidate looks
    like smoke.
    """

    def __init__(
        self,
        groups: list[tuple[np.ndarray, float]],
        *,
        input_size: tuple[int, int] = (50, 50),
        augment: bool = False,
        augment_config=None,
        seed: int = 0,
    ) -> None:
        self.groups = groups
        self.input_size = input_size
        self.augment = augment
        self.augment_config = augment_config
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        from ..data.augment import augment_crop
        from ..models.spatial_cnn import preprocess_crops

        crops, label = self.groups[index]
        if self.augment and self.augment_config is not None:
            crops = np.stack([augment_crop(crop, self.rng, self.augment_config) for crop in crops])
        tensor = preprocess_crops(crops, self.input_size)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


def mil_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for :class:`FrameCropDataset`.

    All the crops of a batch are concatenated into a single tensor and the
    frame boundaries are kept in ``group_sizes`` so that the per-frame
    maximum and mean can be computed.
    """
    crops = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch])
    sizes = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
    return {"crops": crops, "labels": labels, "group_sizes": sizes}


def collect_frame_groups(
    videos: Sequence[VideoFeatures],
    *,
    input_size: tuple[int, int] = (50, 50),
) -> list[tuple[np.ndarray, float]]:
    """Group the candidate crops of every frame that contains a candidate.

    Returns:
        A list of ``(crops, frame_label)`` pairs.
    """
    groups: list[tuple[np.ndarray, float]] = []
    for video in videos:
        if video.crops.size == 0:
            continue
        n_frames = max(1, len(video.brightness))
        frame_label = (
            video.frame_label
            if video.frame_label.size >= n_frames
            else np.zeros(n_frames, dtype=np.int8)
        )
        by_frame: dict[int, list[int]] = {}
        for index, frame_index in enumerate(video.cand_frame_index):
            by_frame.setdefault(int(frame_index), []).append(index)
        for frame_index, indices in by_frame.items():
            label = float(frame_label[frame_index]) if frame_index < len(frame_label) else 0.0
            crops = video.crops[indices]
            if crops.shape[0] == 0:
                continue
            groups.append((crops, label))
    return groups


class TemporalSequenceDataset(Dataset):
    """Windows of per-frame features with current/horizon labels."""

    def __init__(
        self,
        sequences: np.ndarray,
        y_now: np.ndarray,
        y_horizon: np.ndarray,
        mask_now: np.ndarray,
        mask_horizon: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.y_now = y_now
        self.y_horizon = y_horizon
        self.mask_now = mask_now
        self.mask_horizon = mask_horizon
        self.groups = groups

    def __len__(self) -> int:
        return int(self.y_now.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": torch.from_numpy(self.sequences[index]),
            "y_now": torch.tensor(self.y_now[index], dtype=torch.float32),
            "y_horizon": torch.tensor(self.y_horizon[index], dtype=torch.float32),
            "mask_now": torch.tensor(self.mask_now[index], dtype=torch.float32),
            "mask_horizon": torch.tensor(self.mask_horizon[index], dtype=torch.float32),
            "group": torch.tensor(self.groups[index], dtype=torch.long),
        }


def build_temporal_dataset(
    videos: Sequence[VideoFeatures],
    feature_matrices: dict[str, np.ndarray],
    *,
    sequence_length: int = 16,
    horizon: int = 8,
    guard_frames: int = 0,
    stride: int = 1,
) -> TemporalSequenceDataset:
    """Assemble the temporal training dataset from a list of videos."""
    sequences: list[np.ndarray] = []
    y_now: list[float] = []
    y_horizon: list[float] = []
    mask_now: list[bool] = []
    mask_horizon: list[bool] = []
    groups: list[int] = []

    for group_id, video in enumerate(videos):
        features = feature_matrices[video.video_id]
        labels = frame_labels_for_video(video, horizon=horizon, guard_frames=guard_frames)
        n = features.shape[0]
        if n < sequence_length:
            continue
        for end in range(sequence_length - 1, n, max(1, stride)):
            sequences.append(features[end - sequence_length + 1 : end + 1])
            y_now.append(float(labels["y_now"][end]))
            y_horizon.append(float(labels["y_horizon"][end]))
            mask_now.append(bool(labels["mask_now"][end]))
            mask_horizon.append(bool(labels["mask_horizon"][end]))
            groups.append(group_id)

    if not sequences:
        return TemporalSequenceDataset(
            np.zeros((0, sequence_length, FEATURE_DIM), np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, bool),
            np.zeros(0, bool),
            np.zeros(0, np.int64),
        )

    return TemporalSequenceDataset(
        np.asarray(sequences, dtype=np.float32),
        np.asarray(y_now, dtype=np.float32),
        np.asarray(y_horizon, dtype=np.float32),
        np.asarray(mask_now, dtype=bool),
        np.asarray(mask_horizon, dtype=bool),
        np.asarray(groups, dtype=np.int64),
    )


def fit_feature_scaler(dataset: TemporalSequenceDataset) -> FeatureScaler:
    """Fit the feature standardiser on the training windows only."""
    if len(dataset) == 0:
        return FeatureScaler()
    flat = dataset.sequences.reshape(-1, dataset.sequences.shape[-1])
    return FeatureScaler().fit(flat)
