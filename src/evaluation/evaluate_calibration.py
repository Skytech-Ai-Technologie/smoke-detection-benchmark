"""Calibration evaluation.

Reports the expected / maximum calibration error and the Brier score, both
overall and for the subgroups required by the benchmark protocol:

* positive frames and negative frames,
* positive videos and negative videos,
* bright scenes and dark scenes,
* low-motion and high-motion scenes.

The scene subgroups are split at the median of the observed distribution, so
that the split is defined by the data rather than by a hand-picked constant.

Example
-------
    python -m src.evaluation.evaluate_calibration \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn_calibrated/model.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..data.extract_frames import load_features
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .common import add_common_arguments, resolve_config, write_result
from .metrics import calibration_metrics
from .runner import load_specs_from_manifest, run_detector

__all__ = ["main"]

LOGGER = get_logger("evaluation.evaluate_calibration")


def _subgroup_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    if labels.size == 0:
        return {"n": 0, "ece": float("nan"), "mce": float("nan"), "brier": float("nan")}
    metrics = calibration_metrics(labels, probs, n_bins=n_bins)
    metrics.pop("bins", None)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Calibration evaluation.")
    add_common_arguments(parser)
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)
    n_bins = config.get("benchmark", {}).get("calibration_bins", 10)

    specs = load_specs_from_manifest(
        args.manifest,
        args.feature_dir,
        role=args.role,
        split=args.split,
        splits_file=args.splits_file,
    )
    if not specs:
        raise RuntimeError("no videos selected for evaluation")

    results = run_detector(
        specs,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        policy_overrides=args.policy_overrides,
        score_source=args.score_source,
    )

    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    brightness_all: list[np.ndarray] = []
    motion_all: list[np.ndarray] = []
    video_label_all: list[np.ndarray] = []

    for result in results:
        if result.frame_scores.size == 0:
            continue
        labels_all.append(result.frame_labels)
        probs_all.append(result.frame_scores)
        video_label_all.append(
            np.full(result.frame_scores.size, 1 if result.spec.is_positive else 0, dtype=np.int64)
        )

        brightness = np.zeros(result.frame_scores.size, dtype=np.float64)
        motion = np.zeros(result.frame_scores.size, dtype=np.float64)
        archive = Path(args.feature_dir) / f"{result.spec.video_id}.npz"
        if archive.exists():
            data = load_features(archive)
            b = data.get("frame_brightness", np.zeros(0))
            m = data.get("frame_motion", np.zeros(0))
            n = min(result.frame_scores.size, len(b))
            brightness[:n] = b[:n]
            motion[:n] = m[:n]
        brightness_all.append(brightness)
        motion_all.append(motion)

    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    video_labels = np.concatenate(video_label_all)
    brightness = np.concatenate(brightness_all)
    motion = np.concatenate(motion_all)

    overall = calibration_metrics(labels, probs, n_bins=n_bins)
    overall_bins = overall.pop("bins", [])

    brightness_median = float(np.median(brightness)) if brightness.size else 0.0
    motion_median = float(np.median(motion)) if motion.size else 0.0

    subgroups = {
        "positive_frames": _subgroup_metrics(labels[labels == 1], probs[labels == 1], n_bins),
        "negative_frames": _subgroup_metrics(labels[labels == 0], probs[labels == 0], n_bins),
        "positive_videos": _subgroup_metrics(
            labels[video_labels == 1], probs[video_labels == 1], n_bins
        ),
        "negative_videos": _subgroup_metrics(
            labels[video_labels == 0], probs[video_labels == 0], n_bins
        ),
        "bright_scenes": _subgroup_metrics(
            labels[brightness >= brightness_median], probs[brightness >= brightness_median], n_bins
        ),
        "dark_scenes": _subgroup_metrics(
            labels[brightness < brightness_median], probs[brightness < brightness_median], n_bins
        ),
        "high_motion_scenes": _subgroup_metrics(
            labels[motion >= motion_median], probs[motion >= motion_median], n_bins
        ),
        "low_motion_scenes": _subgroup_metrics(
            labels[motion < motion_median], probs[motion < motion_median], n_bins
        ),
    }

    payload = {
        "split": args.split,
        "role": args.role,
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "n_frames": int(labels.size),
        "n_bins": n_bins,
        "brightness_median": brightness_median,
        "motion_median": motion_median,
        "overall": overall,
        "reliability_bins": overall_bins,
        "subgroups": subgroups,
    }

    name = args.tag or experiment_id("calibration_eval", split=args.split, seed=args.seed)
    write_result(args.out_dir, name, payload)

    record = build_record(
        exp_id=name,
        config={"split": args.split, "role": args.role},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        split=args.split,
        model="calibration_eval",
        notes="calibration evaluation",
        metrics=overall,
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "calibration | ECE=%.4f MCE=%.4f Brier=%.4f (n=%d frames)",
        overall.get("ece", float("nan")),
        overall.get("mce", float("nan")),
        overall.get("brier", float("nan")),
        labels.size,
    )
    for name_, values in subgroups.items():
        LOGGER.info("  %-18s n=%-8d ece=%.4f", name_, values.get("n", 0), values.get("ece", float("nan")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
