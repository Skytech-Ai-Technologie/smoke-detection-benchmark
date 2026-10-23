"""Sweep the alarm operating point and compare models at matched recall.

The primary comparison of the benchmark is the one registered in hypothesis
H1: *a calibrated temporal model reduces the false alarms per hour compared
with the original fixed-threshold system **at the same video-level recall***.

That comparison does not depend on one arbitrary operating point: for every
model the script sweeps the confidence threshold on the evaluation videos and
reports the false alarms per hour as a function of the video recall. The
models are then compared by interpolating every curve at a common set of
recall levels.

The sweep is a *reporting* tool: the operating point that the headline
numbers use is the one selected on the validation split by
``src.training.select_thresholds`` and is never taken from this sweep.

Example
-------
    python -m src.evaluation.sweep_curve \\
        --models spatial=none tcn=results/raw/tcn_split01_seed42/model_calibrated.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --split split01 --role test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models.builder import build_detector
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .metrics import aggregate_video_metrics, evaluate_video_events
from .runner import load_specs_from_manifest
from ..training.select_thresholds import _replay_policy

__all__ = ["sweep_model", "compare_at_matched_recall", "main"]

LOGGER = get_logger("evaluation.sweep_curve")

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
DEFAULT_PERSISTENCE = (0.0, 0.5, 1.0, 1.5, 2.0)


def sweep_model(
    name: str,
    specs: Sequence,
    *,
    spatial_checkpoint: str | None,
    temporal_checkpoint: str | None,
    device: str = "cpu",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    persistences: Sequence[float] = DEFAULT_PERSISTENCE,
    refractory_seconds: float = 2.0,
    hysteresis_ratio: float = 0.6,
    mode: str = "fixed",
    variant: str | None = None,
) -> dict[str, Any]:
    """Sweep the threshold of one model and return its operating curve."""
    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides={},
        fps=25.0,
        variant=variant,
    )
    detector = bundle.detector

    streams: list[dict[str, Any]] = []
    for spec in specs:
        detector.reset(spec.video_id)
        result = detector.process_video(spec.path, video_id=spec.video_id)
        streams.append(
            {
                "video_id": spec.video_id,
                "label": spec.label,
                "fps": result.fps,
                "onset_frame": spec.onset_frame,
                "probability": np.asarray(result.temporal_probabilities, dtype=np.float64),
                "uncertainty": np.asarray([d.uncertainty for d in result.decisions], dtype=np.float64),
                "reliability": np.asarray(
                    [d.motion_reliability for d in result.decisions], dtype=np.float64
                ),
            }
        )

    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        for persistence in persistences:
            evaluations = []
            for stream in streams:
                mask = _replay_policy(
                    stream["probability"],
                    stream["uncertainty"],
                    stream["reliability"],
                    threshold=threshold,
                    persistence_seconds=persistence,
                    fps=stream["fps"],
                    refractory_seconds=refractory_seconds,
                    hysteresis_ratio=hysteresis_ratio,
                    mode=mode,
                    max_persistence_seconds=3.0,
                )
                evaluations.append(
                    evaluate_video_events(
                        stream["video_id"],
                        stream["label"],
                        stream["fps"],
                        mask,
                        stream["onset_frame"],
                        refractory_seconds=refractory_seconds,
                    )
                )
            metrics = aggregate_video_metrics(evaluations, refractory_seconds=refractory_seconds)
            points.append(
                {
                    "model": name,
                    "threshold": float(threshold),
                    "persistence_seconds": float(persistence),
                    "video_recall": metrics["video_recall"],
                    "video_specificity": metrics["video_specificity"],
                    "false_alarms_per_hour": metrics["false_alarms_per_hour"],
                    "false_alarm_events": metrics["false_alarm_events"],
                    "delay_mean": metrics.get("delay_mean", float("nan")),
                    "delay_p95": metrics.get("delay_p95", float("nan")),
                }
            )

    # Pareto front: for every recall level keep the lowest false-alarm rate.
    front: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda p: (-p["video_recall"], p["false_alarms_per_hour"])):
        if not front or point["false_alarms_per_hour"] < front[-1]["false_alarms_per_hour"] - 1e-12:
            front.append(point)
    return {"model": name, "points": points, "pareto_front": front}


def compare_at_matched_recall(
    curves: dict[str, dict[str, Any]],
    recall_levels: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> list[dict[str, Any]]:
    """Interpolate the false-alarm rate of every model at a common recall."""
    rows: list[dict[str, Any]] = []
    for level in recall_levels:
        entry: dict[str, Any] = {"recall_level": float(level)}
        for name, curve in curves.items():
            front = curve["pareto_front"]
            if not front:
                entry[name] = float("nan")
                continue
            recalls = np.asarray([p["video_recall"] for p in front], dtype=np.float64)
            fa = np.asarray([p["false_alarms_per_hour"] for p in front], dtype=np.float64)
            order = np.argsort(recalls)
            recalls, fa = recalls[order], fa[order]
            if level < recalls.min() - 1e-9:
                entry[name] = float("nan")
            else:
                entry[name] = float(np.interp(level, recalls, fa))
        rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sweep the operating point of several models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="model specifications of the form name=temporal_checkpoint (use 'none' for no temporal model)",
    )
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--split", default="split01")
    parser.add_argument("--role", default="test")
    parser.add_argument("--manifest", default="data/manifests/firesense_videos.csv")
    parser.add_argument("--feature-dir", default="data/processed/features")
    parser.add_argument("--splits-file", default="data/manifests/splits.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    device = args.device or ("cuda" if _cuda() else "cpu")

    specs = load_specs_from_manifest(
        args.manifest,
        args.feature_dir,
        role=args.role,
        split=args.split,
        splits_file=args.splits_file,
    )
    if not specs:
        raise RuntimeError(f"no videos for role={args.role} in {args.split}")

    curves: dict[str, dict[str, Any]] = {}
    for spec in args.models:
        name, _, rest = spec.partition("=")
        if not name:
            continue
        temporal, _, variant = rest.partition("@")
        LOGGER.info("sweeping model %s (variant=%s)", name, variant or "default")
        curves[name] = sweep_model(
            name,
            specs,
            spatial_checkpoint=args.spatial_checkpoint,
            temporal_checkpoint=None if temporal in ("none", "") else temporal,
            device=device,
            mode=args.mode,
            variant=variant or None,
        )

    comparison = compare_at_matched_recall(curves)
    name = args.tag or experiment_id("sweep", split=args.split, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "split": args.split,
                "role": args.role,
                "mode": args.mode,
                "device": device,
                "curves": curves,
                "matched_recall_comparison": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("wrote %s", path)

    record = build_record(
        exp_id=name,
        config={"split": args.split, "role": args.role, "mode": args.mode},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=device,
        precision="fp32",
        command=" ".join(sys.argv),
        split=args.split,
        model="sweep",
        notes="operating-point sweep for the matched-recall comparison",
        metrics={row["recall_level"]: row for row in comparison},
    )
    append_to_registry(args.registry, record)

    print(json.dumps(comparison, indent=2))
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
