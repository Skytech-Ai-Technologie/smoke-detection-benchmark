"""Offline frame-level evaluation.

Reports the frame-level detection quality of a model: precision, recall, F1,
balanced accuracy, Matthews correlation, AUROC and the area under the
precision-recall curve.

The frame labels are derived from the smoke onset of each video: a frame is
positive when the smoke onset has already been reached.

Example
-------
    python -m src.evaluation.evaluate_frame \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn/model.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --split split01 --role test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .common import add_common_arguments, resolve_config, write_result
from .metrics import calibration_metrics, frame_metrics
from .runner import load_specs_from_manifest, run_detector

__all__ = ["main"]

LOGGER = get_logger("evaluation.evaluate_frame")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Offline frame-level evaluation.")
    add_common_arguments(parser)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)

    threshold = args.threshold
    if threshold is None:
        threshold = config.get("benchmark", {}).get("frame_threshold", 0.5)

    specs = load_specs_from_manifest(
        args.manifest,
        args.feature_dir,
        role=args.role,
        split=args.split,
        splits_file=args.splits_file,
    )
    if not specs:
        raise RuntimeError("no videos selected for evaluation")
    LOGGER.info("evaluating %d videos (%s / %s)", len(specs), args.split, args.role)

    results = run_detector(
        specs,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        policy_overrides=args.policy_overrides,
        score_source=args.score_source,
        variant=args.variant,
    )

    scores = np.concatenate([r.frame_scores for r in results])
    labels = np.concatenate([r.frame_labels for r in results])

    metrics = frame_metrics(labels, scores, threshold=threshold)
    calibration = calibration_metrics(
        labels, scores, n_bins=config.get("benchmark", {}).get("calibration_bins", 10)
    )

    # Per-video frame metrics (the unit of analysis for the statistics).
    per_video: list[dict[str, Any]] = []
    for result in results:
        if result.frame_scores.size == 0:
            continue
        video_metrics = frame_metrics(result.frame_labels, result.frame_scores, threshold=threshold)
        per_video.append({"video_id": result.spec.video_id, "label": result.spec.label, **video_metrics})

    payload = {
        "split": args.split,
        "role": args.role,
        "score_source": args.score_source,
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "n_videos": len(results),
        "frame_metrics": metrics,
        "calibration": calibration,
        "per_video": per_video,
    }

    name = args.tag or experiment_id("frame_eval", split=args.split, seed=args.seed)
    out_path = write_result(args.out_dir, name, payload)

    # The raw per-frame scores are stored so that the plotting scripts can
    # build ROC / PR / reliability curves without re-running the evaluation.
    npz_path = Path(args.out_dir) / f"{name}_frames.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        scores=scores,
        labels=labels,
        video_ids=np.array([r.spec.video_id for r in results]),
    )

    csv_path = Path(args.out_dir) / f"{name}_per_video.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if per_video:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_video[0].keys()))
            writer.writeheader()
            for row in per_video:
                writer.writerow(row)
        LOGGER.info("wrote %s", csv_path)

    record = build_record(
        exp_id=name,
        config={"score_source": args.score_source, "split": args.split, "role": args.role},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        split=args.split,
        model=f"frame_eval/{args.score_source}",
        notes="offline frame evaluation",
        metrics=metrics,
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "frame metrics | precision=%.4f recall=%.4f f1=%.4f auroc=%.4f auprc=%.4f ece=%.4f",
        metrics.get("precision", float("nan")),
        metrics.get("recall", float("nan")),
        metrics.get("f1", float("nan")),
        metrics.get("auroc", float("nan")),
        metrics.get("auprc", float("nan")),
        calibration.get("ece", float("nan")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
