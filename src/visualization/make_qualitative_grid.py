"""Qualitative figure: Kalman/HSV candidate masks, temporal curves, decision log.

For one or more selected videos the script produces a grid with

* the raw frame,
* the Kalman foreground mask,
* the HSV candidate mask with the detected candidate boxes,
* the temporal probability curves, the **decision log** (temporal
  probability vs. the alarm threshold actually applied per frame, with the
  alarm intervals shaded) and the binary alarm state,

which is the qualitative counterpart of the quantitative tables. Accepted via
``--video`` several times to compare a positive and, e.g., a negative video
in one figure; onsets are read automatically from
``data/annotations/onsets.csv`` unless ``--onset-frame`` is given.

Example
-------
    python -m src.visualization.make_qualitative_grid \\
        --model results/raw/tcn_split01_seed42/model.pt \\
        --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt \\
        --video data/raw/firesense_smoke/pos/testpos10.826.avi \\
        --video data/raw/firesense_smoke/neg/testneg02.808.avi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger, setup_logging

__all__ = ["build_qualitative_grid", "main"]

LOGGER = get_logger("visualization.make_qualitative_grid")


def _select_frames(onset: int | None, n_frames: int, count: int = 4) -> list[int]:
    if onset is None or onset <= 0:
        return [int(i * n_frames / (count + 1)) for i in range(1, count + 1)]
    offsets = [-0.5, 0.0, 0.5, 1.5]
    picks: list[int] = []
    for offset in offsets:
        index = int(round(onset + offset * 10))
        picks.append(min(max(index, 0), n_frames - 1))
    return picks


def _auto_onset(video_path: str, manual: dict[str, dict[str, Any]]) -> int | None:
    """Onset frame of a video from the annotation CSV, ``None`` if absent."""
    name = Path(video_path).name
    annotation = manual.get(name) or manual.get(name.replace(".avi", ""))
    if not annotation:
        return None
    onset = annotation.get("onset_frame")
    return int(onset) if onset is not None else None


def _role(video_path: str, onset: int | None) -> str:
    """Describe the video's role for the figure label."""
    stem = Path(video_path).name
    if "neg" in stem.lower():
        return "negative — expect no alarm"
    is_pos = "pos" in stem.lower()
    if is_pos and onset is None:
        return "positive — onset ?"
    return f"positive — smoke onset frame {onset}" if is_pos else ""


def _run_video(
    video_path: str,
    detector: Any,
    fps: float,
    onset: int | None,
) -> dict[str, Any]:
    """Decode one video, pick frames and run the full detector over it."""
    import cv2

    from ..models.kalman_hsv import (
        HSVCandidateConfig,
        KalmanBackgroundSubtractor,
        KalmanConfig,
        extract_candidates,
        smoke_likelihood_mask,
    )

    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")

    picks = _select_frames(onset, len(frames))

    # Re-run the front end on the selected frames to obtain the masks.
    kalman = KalmanBackgroundSubtractor(KalmanConfig())
    hsv_cfg = HSVCandidateConfig()
    masks: dict[int, tuple[np.ndarray, np.ndarray, list]] = {}
    for i, frame in enumerate(frames):
        fg, residual = kalman.apply(frame)
        if i in picks:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            candidate_mask = smoke_likelihood_mask(hsv, hsv_cfg, fg)
            candidates = extract_candidates(frame, i, fg, residual, hsv_cfg)
            masks[i] = (fg, candidate_mask, candidates)

    # Run the detector to obtain the decision log.
    detector.reset("qualitative")
    curves: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        record = detector.process_frame(frame, i, timestamp=i / fps)
        curves.append(record.to_dict())

    return {"frames": frames, "picks": picks, "masks": masks, "curves": curves, "onset": onset}


def _clean(values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (finite mask, values as float array) for possibly-None series."""
    arr = np.asarray(
        [float(v) if v is not None else np.nan for v in values], dtype=np.float64
    )
    finite = np.isfinite(arr)
    return arr, finite


def build_qualitative_grid(
    video_path: str | list[str],
    *,
    spatial_checkpoint: str | None,
    temporal_checkpoint: str | None,
    device: str = "cpu",
    onset_frame: int | None = None,
    out_path: str | Path = "results/figures/qualitative.png",
    fps: float = 25.0,
    annotations: str | Path = "data/annotations/onsets.csv",
) -> Path:
    """Render the qualitative grid, one block of three rows per video."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..models.builder import build_detector

    if isinstance(video_path, (str, Path)):
        video_paths = [str(video_path)]
    else:
        video_paths = [str(path) for path in video_path]

    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides={},
        fps=fps,
    )
    detector = bundle.detector

    # Onset resolution: explicit CLI value wins (applies to every video),
    # otherwise the annotation CSV, otherwise nothing.
    manual: dict[str, dict[str, Any]] = {}
    from ..data.extract_frames import load_manual_onsets

    manual = load_manual_onsets(annotations)

    blocks: list[dict[str, Any]] = []
    for path in video_paths:
        onset = onset_frame if onset_frame is not None else _auto_onset(path, manual)
        LOGGER.info(
            "processing %s (onset=%s)", path, onset if onset is not None else "none"
        )
        run = _run_video(path, detector, fps, onset)
        blocks.append(
            {
                "run": run,
                "label": Path(path).name,
                "role": _role(path, onset),
            }
        )

    n_cols = len(blocks[0]["run"]["picks"]) + 1
    n_rows = 3 * len(blocks)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * len(blocks) + 1.2)
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] != n_rows:  # single-row figure edge case
        axes = axes.reshape(n_rows, n_cols)

    for block_idx, block in enumerate(blocks):
        base = 3 * block_idx
        run = block["run"]
        frames = run["frames"]
        picks = run["picks"]
        masks = run["masks"]
        curves = run["curves"]
        onset = run["onset"]

        for col, index in enumerate(picks):
            frame = frames[index]
            axes[base, col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if block_idx == 0:
                axes[base, col].set_title(f"t={index} ({index / fps:.1f}s)", fontsize=8)

            if index in masks:
                fg, _candidate_mask, candidates = masks[index]
                axes[base + 1, col].imshow(fg, cmap="gray", vmin=0, vmax=255)
                overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                for candidate in candidates:
                    x0, y0, x1, y1 = candidate.bbox_xyxy
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                axes[base + 2, col].imshow(overlay)
                if block_idx == 0:
                    axes[base + 1, col].set_title("Kalman foreground", fontsize=8)
                    axes[base + 2, col].set_title(
                        f"HSV candidates (n={len(candidates)})", fontsize=8
                    )
            for row in range(3):
                axes[base + row, col].set_xticks([])
                axes[base + row, col].set_yticks([])

        time_axis = np.array([c["timestamp"] for c in curves])
        temporal, temporal_finite = _clean([c["temporal_probability"] for c in curves])
        calibrated, calibrated_finite = _clean(
            [c["calibrated_probability"] for c in curves]
        )
        frame_p, frame_finite = _clean([c["frame_probability"] for c in curves])
        thresholds = np.asarray([c["threshold"] for c in curves], dtype=np.float64)
        alarm = np.asarray([c["alarm_state"] == "alarm" for c in curves], dtype=float)

        # Video label on the left of each block.
        axes[base, 0].set_ylabel(
            f"{block['label']}\n{block['role']}", fontsize=7, rotation=0,
            ha="right", va="center", labelpad=6,
        )

        # Row 0: probability curves.
        ax = axes[base, -1]
        ax.fill_between(time_axis, 0, 1, where=alarm > 0, color="crimson", alpha=0.12)
        if temporal_finite.any():
            ax.plot(time_axis[temporal_finite], temporal[temporal_finite], label="temporal p")
        if calibrated_finite.any():
            ax.plot(
                time_axis[calibrated_finite],
                calibrated[calibrated_finite],
                alpha=0.6,
                linewidth=0.9,
                label="calibrated p",
            )
        if frame_finite.any():
            ax.plot(time_axis[frame_finite], frame_p[frame_finite], alpha=0.35,
                    linewidth=0.8, label="frame p")
        if onset is not None:
            ax.axvline(onset / fps, color="green", linestyle="--", linewidth=1,
                       label="onset")
        ax.set_ylim(-0.02, 1.05)
        if block_idx == 0:
            ax.legend(fontsize=6, loc="upper left")
            ax.set_title("probabilities", fontsize=8)

        # Row 1: decision log for neural models, raw evidence otherwise.
        ax = axes[base + 1, -1]
        if bundle.temporal_kind in ("tcn", "gru"):
            ax.fill_between(time_axis, 0, 1, where=alarm > 0, color="crimson", alpha=0.12)
            if temporal_finite.any():
                ax.plot(time_axis[temporal_finite], temporal[temporal_finite],
                        label="temporal p")
            ax.plot(time_axis, thresholds, color="black", linestyle=":", linewidth=1.2,
                    label="alarm threshold")
            if block_idx == 0:
                ax.set_title("decision log (p vs threshold)", fontsize=8)
                ax.legend(fontsize=6, loc="upper left")
        else:
            evidence, evidence_finite = _clean([c["temporal_evidence"] for c in curves])
            if evidence_finite.any():
                ax.plot(time_axis[evidence_finite], evidence[evidence_finite], color="C3")
            # Evidence-accumulator alarm/cancel levels (unchanged behaviour).
            ax.axhline(8.0, color="k", linestyle=":", linewidth=0.8)
            ax.axhline(-4.0, color="k", linestyle=":", linewidth=0.8)
            if block_idx == 0:
                ax.set_title("evidence", fontsize=8)
        if bundle.temporal_kind not in ("tcn", "gru"):
            ax.set_ylim(-0.02, 1.05)

        # Row 2: binary alarm state.
        ax = axes[base + 2, -1]
        ax.fill_between(time_axis, 0, alarm, step="post", color="crimson", alpha=0.5)
        ax.set_ylim(0, 1.2)
        if block_idx == 0:
            ax.set_title("alarm state", fontsize=8)
        for row in range(3):
            axes[base + row, -1].tick_params(labelsize=6)

    fig.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    LOGGER.info("wrote %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the qualitative figure.")
    parser.add_argument("--video", required=True, nargs="+", action="extend",
                        help="video path(s); repeat --video to add more blocks")
    parser.add_argument("--model", default=None)
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--onset-frame", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--out", default="results/figures/qualitative.png")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    build_qualitative_grid(
        args.video,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        onset_frame=args.onset_frame,
        out_path=args.out,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
