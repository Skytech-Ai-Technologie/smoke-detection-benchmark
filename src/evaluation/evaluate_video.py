"""Offline video-level evaluation.

This is the primary evaluation of the project: it reports the video-level
safety metrics (video recall, video specificity, false alarms per hour,
detection delay, recall within 0.5/1.0/2.0 s) together with the per-video
records, which are the unit of analysis for every statistical test.

Example
-------
    python -m src.evaluation.evaluate_video \\
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
from .metrics import aggregate_video_metrics, evaluate_video_events
from .runner import load_specs_from_manifest, run_detector

__all__ = ["main"]

LOGGER = get_logger("evaluation.evaluate_video")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Offline video-level evaluation.")
    add_common_arguments(parser)
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)

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

    evaluations = []
    for result in results:
        mask = np.array([d.alarm_state == "alarm" for d in result.decisions], dtype=bool)
        evaluations.append(
            evaluate_video_events(
                result.spec.video_id,
                result.spec.label,
                result.fps,
                mask,
                result.spec.onset_frame,
                refractory_seconds=args.refractory_seconds,
            )
        )

    metrics = aggregate_video_metrics(evaluations, refractory_seconds=args.refractory_seconds)
    per_video = [e.to_dict() for e in evaluations]

    payload = {
        "split": args.split,
        "role": args.role,
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "policy": args.policy_overrides,
        "refractory_seconds": args.refractory_seconds,
        "video_metrics": metrics,
        "per_video": per_video,
    }

    name = args.tag or experiment_id("video_eval", split=args.split, seed=args.seed)
    write_result(args.out_dir, name, payload)

    csv_path = Path(args.out_dir) / f"{name}_per_video.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        for row in per_video:
            writer.writerow(row)
    LOGGER.info("wrote %s", csv_path)

    record = build_record(
        exp_id=name,
        config={"split": args.split, "role": args.role, "policy": args.policy_overrides},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        split=args.split,
        model="video_eval",
        notes="offline video evaluation",
        metrics=metrics,
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "video metrics | recall=%.3f specificity=%.3f fa/h=%.3f delay_mean=%.2fs delay_p95=%.2fs",
        metrics["video_recall"],
        metrics["video_specificity"],
        metrics["false_alarms_per_hour"],
        metrics.get("delay_mean", float("nan")),
        metrics.get("delay_p95", float("nan")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
