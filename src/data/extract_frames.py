"""Extract and cache candidate features for every video.

Running the Kalman/HSV front end is by far the most expensive part of the
pipeline, so the results are cached once per video in
``data/processed/features/<video_id>.npz`` and reused by every experiment.

Two kinds of data are stored:

* **frame level** -- brightness, blur, foreground ratio, candidate count and
  the per-frame motion energy, one value per frame;
* **candidate level** -- bounding box, area, mean HSV, motion magnitude, the
  50x50 crop used by the spatial CNN, and a three-valued label
  (1 = smoke, 0 = background, -1 = ignore).

The candidate label is derived from the annotated smoke onset of the video
whenever an annotation exists; otherwise the documented weak-onset heuristic
in :func:`estimate_onset` is used and the video is flagged accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..models.kalman_hsv import (
    HSVCandidateConfig,
    KalmanBackgroundSubtractor,
    KalmanConfig,
    extract_candidates,
    image_quality,
)
from ..utils.logging import get_logger, setup_logging

__all__ = [
    "ExtractionConfig",
    "estimate_onset",
    "extract_video_features",
    "extract_dataset",
    "load_features",
    "main",
]

LOGGER = get_logger("data.extract_frames")

LABEL_SMOKE = 1
LABEL_BACKGROUND = 0
LABEL_IGNORE = -1


def load_manual_onsets(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the manual onset annotations.

    The file is a CSV with the columns ``video_name``, ``onset_frame``,
    ``source``, ``confidence`` and (optionally) ``official_onset_frame``.

    Returns:
        A mapping ``video_file_name -> annotation record``.
    """
    import csv

    out: dict[str, dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return out
    # The annotation file may contain comment lines before the header.
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # Be robust to a header that is preceded by other content.
    for index, line in enumerate(lines):
        if line.lower().startswith("video_name"):
            lines = lines[index:]
            break
    for row in csv.DictReader(lines):
            name = str(row.get("video_name", "")).strip()
            if not name:
                continue
            try:
                onset = int(float(str(row.get("onset_frame", "")).strip()))
            except (TypeError, ValueError):
                continue
            official = str(row.get("official_onset_frame", "")).strip()
            try:
                official_frame: int | None = int(float(official)) if official else None
            except ValueError:
                official_frame = None
            out[name] = {
                "onset_frame": onset,
                "source": str(row.get("source", "manual")).strip(),
                "confidence": str(row.get("confidence", "")).strip(),
                "official_onset_frame": official_frame,
                "notes": str(row.get("notes", "")).strip(),
            }
    return out


@dataclass
class ExtractionConfig:
    """Parameters of the front-end extraction."""

    kalman: dict[str, Any] | None = None
    hsv: dict[str, Any] | None = None
    crop_size: int = 50
    stride: int = 1
    pre_onset_guard_frames: int = 0
    weak_onset_area_ratio: float = 0.010
    weak_onset_window_seconds: float = 2.0
    weak_onset_support: float = 0.50
    write_crops: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_onset(
    frame_area_ratio: np.ndarray,
    fps: float,
    config: ExtractionConfig | None = None,
) -> int | None:
    """Weak onset estimator used when the dataset has no onset annotation.

    The onset is the first frame at which a low-saturation candidate covers
    at least ``weak_onset_area_ratio`` of the frame and stays visible for at
    least ``weak_onset_support`` of the following ``weak_onset_window_seconds``.

    Returns:
        The estimated onset frame, or ``None`` when no onset is found.
    """
    cfg = config or ExtractionConfig()
    if frame_area_ratio.size == 0:
        return None
    above = frame_area_ratio >= cfg.weak_onset_area_ratio
    if not above.any():
        return None

    window = max(1, int(round(cfg.weak_onset_window_seconds * max(fps, 1.0))))
    n = len(above)
    for i in range(n):
        end = min(n, i + window)
        if end - i < max(2, window // 4):
            break
        support = float(above[i:end].mean())
        if support >= cfg.weak_onset_support:
            return int(i)
    return None


def _candidate_labels(
    label: str,
    onset_frame: int | None,
    frame_index: int,
    pre_onset_guard: int,
) -> int:
    """Three-valued label for a candidate observed at ``frame_index``."""
    if label != "smoke":
        return LABEL_BACKGROUND
    if onset_frame is None:
        return LABEL_IGNORE
    if frame_index >= onset_frame:
        return LABEL_SMOKE
    if frame_index >= onset_frame - pre_onset_guard:
        return LABEL_IGNORE
    return LABEL_BACKGROUND


def extract_video_features(
    video_path: str | Path,
    *,
    video_id: str = "",
    label: str = "unknown",
    onset_frame: int | None = None,
    config: ExtractionConfig | None = None,
) -> dict[str, Any]:
    """Run the front end over one video and return the cached arrays."""
    cfg = config or ExtractionConfig()
    kalman_cfg = KalmanConfig(**(cfg.kalman or {}))
    hsv_cfg = HSVCandidateConfig(**(cfg.hsv or {}))
    if cfg.crop_size:
        hsv_cfg.crop_size = int(cfg.crop_size)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    kalman = KalmanBackgroundSubtractor(kalman_cfg)

    brightness: list[float] = []
    blur: list[float] = []
    fg_ratio: list[float] = []
    frame_motion: list[float] = []
    max_area_ratio: list[float] = []

    cand_frame: list[int] = []
    cand_bbox: list[tuple[int, int, int, int]] = []
    cand_area: list[int] = []
    cand_hsv: list[tuple[float, float, float]] = []
    cand_motion: list[float] = []
    crops: list[np.ndarray] = []

    index = 0
    stride = max(1, int(cfg.stride))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            fg_mask, residual = kalman.apply(frame)
            candidates = extract_candidates(
                frame, index, foreground_mask=fg_mask, residual=residual, config=hsv_cfg
            )
            b, bl = image_quality(frame)
            brightness.append(b)
            blur.append(bl)
            fg_ratio.append(float((fg_mask > 0).mean()))
            frame_motion.append(float(residual.mean()) if residual.size else 0.0)

            frame_area = max(1, frame.shape[0] * frame.shape[1])
            areas = [c.area / frame_area for c in candidates]
            max_area_ratio.append(float(max(areas)) if areas else 0.0)

            for cand in candidates:
                cand_frame.append(index)
                cand_bbox.append(cand.bbox_xyxy)
                cand_area.append(cand.area)
                cand_hsv.append(cand.mean_hsv)
                cand_motion.append(cand.motion_magnitude)
                if cfg.write_crops and cand.crop is not None:
                    crops.append(cand.crop)
        index += 1
    capture.release()

    onset_source = "annotation"
    if onset_frame is None and label == "smoke":
        onset_frame = estimate_onset(np.asarray(max_area_ratio, dtype=np.float64), fps, cfg)
        onset_source = "heuristic"
    elif onset_frame is None:
        onset_source = "not_applicable"

    # The candidate labels are assigned *after* the onset has been resolved,
    # because the weak-onset estimator needs the candidate statistics of the
    # whole video.
    cand_label = [
        _candidate_labels(label, onset_frame, int(f), cfg.pre_onset_guard_frames)
        for f in cand_frame
    ]

    # Frame-level labels. In a positive video a frame is positive from the
    # onset onwards. The frame labels drive the weakly-supervised (MIL)
    # training of the spatial classifier.
    n_frames = max(1, len(brightness))
    frame_label = np.zeros(n_frames, dtype=np.int8)
    if label == "smoke" and onset_frame is not None:
        frame_label[min(int(onset_frame), n_frames) :] = LABEL_SMOKE

    data: dict[str, Any] = {
        "video_id": video_id or Path(video_path).stem,
        "path": str(video_path),
        "label": label,
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": index,
        "onset_frame": np.array(-1 if onset_frame is None else int(onset_frame)),
        "onset_source": onset_source,
        "stride": stride,
        "frame_brightness": np.asarray(brightness, dtype=np.float32),
        "frame_blur": np.asarray(blur, dtype=np.float32),
        "frame_fg_ratio": np.asarray(fg_ratio, dtype=np.float32),
        "frame_motion": np.asarray(frame_motion, dtype=np.float32),
        "frame_max_area_ratio": np.asarray(max_area_ratio, dtype=np.float32),
        "frame_label": frame_label,
        "cand_frame_index": np.asarray(cand_frame, dtype=np.int32),
        "cand_bbox": np.asarray(cand_bbox, dtype=np.int32).reshape(-1, 4),
        "cand_area": np.asarray(cand_area, dtype=np.int32),
        "cand_hsv": np.asarray(cand_hsv, dtype=np.float32).reshape(-1, 3),
        "cand_motion": np.asarray(cand_motion, dtype=np.float32),
        "cand_label": np.asarray(cand_label, dtype=np.int8),
    }
    if cfg.write_crops and crops:
        data["cand_crop"] = np.stack(crops).astype(np.uint8)
    else:
        data["cand_crop"] = np.zeros((0, hsv_cfg.crop_size, hsv_cfg.crop_size, 3), dtype=np.uint8)
    return data


def save_features(data: dict[str, Any], path: str | Path) -> Path:
    """Persist the extracted arrays to a compressed npz archive."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return out


def load_features(path: str | Path) -> dict[str, Any]:
    """Load a cached feature archive."""
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def relabel_features(
    path: str | Path,
    *,
    onset_frame: int | None,
    onset_source: str = "manual",
    pre_onset_guard_frames: int = 0,
    official_onset_frame: int | None = None,
) -> dict[str, Any]:
    """Update the labels of a cached feature archive in place.

    The front end (Kalman + HSV + candidates) is the expensive part of the
    pipeline and does not depend on the annotations, so the labels can be
    refreshed without re-extracting the frames.
    """
    source = Path(path)
    data = load_features(source)
    label = str(data.get("label", "unknown"))
    n_frames = int(len(data.get("frame_brightness", [])))

    frame_label = np.zeros(max(1, n_frames), dtype=np.int8)
    if label == "smoke" and onset_frame is not None:
        frame_label[min(int(onset_frame), max(1, n_frames)) :] = LABEL_SMOKE

    cand_frame = data.get("cand_frame_index", np.zeros(0, np.int32))
    cand_label = np.array(
        [
            _candidate_labels(label, onset_frame, int(f), pre_onset_guard_frames)
            for f in cand_frame
        ],
        dtype=np.int8,
    )

    data["onset_frame"] = np.array(-1 if onset_frame is None else int(onset_frame))
    data["onset_source"] = onset_source
    data["official_onset_frame"] = np.array(
        -1 if official_onset_frame is None else int(official_onset_frame)
    )
    data["frame_label"] = frame_label
    data["cand_label"] = cand_label
    save_features(data, source)
    return {
        "video_id": str(data.get("video_id", "")),
        "onset_frame": onset_frame,
        "onset_source": onset_source,
        "n_frames": n_frames,
        "n_positive_frames": int(frame_label.sum()),
        "n_candidates": int(len(cand_frame)),
    }


def extract_dataset(
    manifest_rows: list[dict[str, Any]],
    out_dir: str | Path,
    config: ExtractionConfig | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract features for every video of a manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"videos": [], "out_dir": str(out)}

    for row in manifest_rows:
        video_id = row["video_id"]
        target = out / f"{video_id}.npz"
        if target.exists() and not overwrite:
            LOGGER.info("%s already extracted; skipping", video_id)
            continue

        onset_raw = row.get("onset_frame", "")
        onset: int | None = None
        if str(onset_raw).strip() not in ("", "None", "nan"):
            try:
                onset = int(float(onset_raw))
            except (TypeError, ValueError):
                onset = None

        data = extract_video_features(
            row["path"],
            video_id=video_id,
            label=row.get("label", "unknown"),
            onset_frame=onset,
            config=config,
        )
        save_features(data, target)
        summary["videos"].append(
            {
                "video_id": video_id,
                "num_frames": int(data["num_frames"]),
                "n_candidates": int(len(data["cand_frame_index"])),
                "onset_frame": int(data["onset_frame"]),
                "onset_source": str(data["onset_source"]),
            }
        )
        LOGGER.info(
            "extracted %s: %d frames, %d candidates, onset=%s (%s)",
            video_id,
            data["num_frames"],
            len(data["cand_frame_index"]),
            int(data["onset_frame"]),
            str(data["onset_source"]),
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract and cache candidate features.")
    parser.add_argument("--manifest", required=True, help="video manifest CSV")
    parser.add_argument("--out-dir", default="data/processed/features", help="cache directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-crops", action="store_true", help="do not store the CNN crops")
    parser.add_argument(
        "--annotations",
        default=None,
        help="manual onset annotation CSV (data/annotations/onsets.csv)",
    )
    parser.add_argument(
        "--relabel-only",
        action="store_true",
        help="only refresh the labels of the existing cache (no re-extraction)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    from .build_manifest import MANIFEST_COLUMNS
    from .video_split import read_manifest

    rows = read_manifest(args.manifest)
    config = ExtractionConfig(stride=args.stride, write_crops=not args.no_crops)

    manual = load_manual_onsets(args.annotations) if args.annotations else {}
    if manual:
        LOGGER.info("loaded %d manual onset annotations from %s", len(manual), args.annotations)

    if args.relabel_only:
        summary = {"videos": []}
        for row in rows:
            video_id = row["video_id"]
            target = Path(args.out_dir) / f"{video_id}.npz"
            if not target.exists():
                LOGGER.warning("%s has no cached features; skipping", video_id)
                continue
            annotation = manual.get(row.get("video_name", ""), {})
            onset = annotation.get("onset_frame")
            if onset is None and str(row.get("label")) == "smoke":
                LOGGER.warning("%s: no manual onset annotation; keeping the cached value", video_id)
                continue
            info = relabel_features(
                target,
                onset_frame=onset,
                onset_source=annotation.get("source", "manual"),
                pre_onset_guard_frames=config.pre_onset_guard_frames,
                official_onset_frame=annotation.get("official_onset_frame"),
            )
            summary["videos"].append(info)
            LOGGER.info("relabelled %s: onset=%s (%s)", video_id, onset, info["onset_source"])
        summary_path = Path(args.out_dir) / "extraction_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", summary_path)
        return 0

    summary = extract_dataset(rows, args.out_dir, config, overwrite=args.overwrite)

    summary_path = Path(args.out_dir) / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
