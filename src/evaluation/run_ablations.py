"""Run the ablation grid.

Every ablation is evaluated on the **same** video split as the main result
and produces per-video records, so that the variants can be compared with
paired statistical tests.

The variants that only change the *inference* path (no Kalman motion model,
no HSV filtering, CNN only, CNN plus fixed persistence, fixed vs adaptive
threshold, frame skipping, hysteresis values) are executed by this script.
The variants that require *retraining* (GRU vs TCN, sequence length 8/16/32/64,
FP32 vs FP16 vs INT8, temperature vs isotonic calibration) are produced by
the training scripts and are collected from ``results/raw`` by
``src.evaluation.make_tables``.

Example
-------
    python -m src.evaluation.run_ablations \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn/model.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --split split01 --role test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..models.calibrated_detector import DetectorConfig
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .common import add_common_arguments, resolve_config, write_result
from .metrics import aggregate_video_metrics, evaluate_video_events
from .runner import load_specs_from_manifest

__all__ = ["ABLATIONS", "run_ablation", "main"]

LOGGER = get_logger("evaluation.run_ablations")

# Inference-time ablations. Each entry overrides the detector configuration
# and/or the alarm policy.
ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {
        "description": "The complete proposed system (causal TCN + adaptive policy).",
    },
    "no_kalman_motion_model": {
        "description": "The Kalman foreground mask does not gate the candidates.",
        "detector": {"use_kalman_foreground": False},
    },
    "no_hsv_filtering": {
        "description": "The low-saturation constraint of the candidate mask is dropped.",
        "detector": {"use_hsv_filter": False},
    },
    "cnn_only": {
        "description": "Frame probability only, no temporal model (thresholded at 0.5).",
        "temporal_kind": "none",
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "cnn_plus_fixed_persistence": {
        "description": "Frame probability with the fixed persistence rule.",
        "temporal_kind": "voting",
        "voting": {"mode": "persistence", "persistence_seconds": 1.5, "confidence_threshold": 0.5},
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "fixed_voting_mean": {
        "description": "Moving average of the frame probability over 3 s.",
        "temporal_kind": "voting",
        "voting": {"mode": "mean", "window_seconds": 3.0, "confidence_threshold": 0.5},
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "fixed_voting_median": {
        "description": "Moving median of the frame probability over 3 s.",
        "temporal_kind": "voting",
        "voting": {"mode": "median", "window_seconds": 3.0, "confidence_threshold": 0.5},
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "fixed_voting_majority": {
        "description": "Majority vote over the frame probabilities of a 3 s window.",
        "temporal_kind": "voting",
        "voting": {"mode": "majority", "window_seconds": 3.0, "confidence_threshold": 0.5},
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "fixed_voting_max": {
        "description": "Moving maximum of the frame probability over 3 s.",
        "temporal_kind": "voting",
        "voting": {"mode": "max", "window_seconds": 3.0, "confidence_threshold": 0.5},
        "policy": {"mode": "fixed", "confidence_threshold": 0.5, "persistence_seconds": 0.0},
    },
    "no_calibration": {
        "description": "The temporal probability is used without calibration.",
        "calibrator": "none",
    },
    "fixed_threshold_policy": {
        "description": "Fixed confidence threshold and fixed persistence.",
        "policy": {"mode": "fixed"},
    },
    "adaptive_threshold_policy": {
        "description": "Adaptive threshold and adaptive required persistence.",
        "policy": {"mode": "adaptive"},
    },
    "adaptive_frame_skipping": {
        "description": "Adaptive frame skipping in the low-risk state.",
        "policy": {"frame_skip": True, "frame_skip_interval": 2},
    },
    "tcn_without_motion_features": {
        "description": "The handcrafted motion features are removed.",
        "detector": {"zero_features": ["motion_magnitude", "area_change", "persistence"]},
    },
    "tcn_without_quality_features": {
        "description": "The image-quality features are removed.",
        "detector": {"zero_features": ["brightness", "blur_score"]},
    },
    "hysteresis_0_3": {
        "description": "Alarm hysteresis ratio 0.3.",
        "policy": {"hysteresis_ratio": 0.3},
    },
    "hysteresis_0_6": {
        "description": "Alarm hysteresis ratio 0.6 (default).",
        "policy": {"hysteresis_ratio": 0.6},
    },
    "hysteresis_0_9": {
        "description": "Alarm hysteresis ratio 0.9.",
        "policy": {"hysteresis_ratio": 0.9},
    },
}


def run_ablation(
    name: str,
    spec: dict[str, Any],
    specs: list,
    *,
    spatial_checkpoint: str | None,
    temporal_checkpoint: str | None,
    device: str,
    policy_overrides: dict[str, Any],
    refractory_seconds: float,
    base_config: DetectorConfig,
) -> dict[str, Any]:
    """Evaluate one ablation variant."""
    from ..models.builder import build_detector

    detector_config = DetectorConfig.from_dict(base_config.to_dict())
    for key, value in (spec.get("detector") or {}).items():
        setattr(detector_config, key, value)
    if "voting" in spec:
        detector_config.voting = {**detector_config.voting, **spec["voting"]}

    merged_policy = dict(policy_overrides)
    merged_policy.update(spec.get("policy") or {})

    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides=merged_policy,
        detector_config=detector_config,
        temporal_kind_override=spec.get("temporal_kind"),
        calibrator_override=spec.get("calibrator"),
    )
    detector = bundle.detector

    evaluations = []
    for video_spec in specs:
        detector.reset(video_spec.video_id)
        result = detector.process_video(video_spec.path, video_id=video_spec.video_id)
        mask = np.array([d.alarm_state == "alarm" for d in result.decisions], dtype=bool)
        evaluations.append(
            evaluate_video_events(
                video_spec.video_id,
                video_spec.label,
                result.fps,
                mask,
                video_spec.onset_frame,
                refractory_seconds=refractory_seconds,
            )
        )

    metrics = aggregate_video_metrics(evaluations, refractory_seconds=refractory_seconds)
    return {
        "ablation": name,
        "description": spec.get("description", ""),
        "video_metrics": metrics,
        "per_video": [e.to_dict() for e in evaluations],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the ablation grid.")
    add_common_arguments(parser)
    parser.add_argument("--ablations", nargs="*", default=None, help="subset of ablation names")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)

    if getattr(args, "policy_file", None):
        payload = json.loads(Path(args.policy_file).read_text(encoding="utf-8"))
        selection = payload.get("selection", payload)
        args.policy_overrides = {
            "mode": selection.get("mode", "fixed"),
            "confidence_threshold": float(selection.get("threshold", 0.7)),
            "persistence_seconds": float(selection.get("persistence_seconds", 0.0)),
        }
        LOGGER.info("ablations use the operating point %s", args.policy_overrides)

    specs = load_specs_from_manifest(
        args.manifest,
        args.feature_dir,
        role=args.role,
        split=args.split,
        splits_file=args.splits_file,
    )
    if not specs:
        raise RuntimeError("no videos selected for the ablation run")
    LOGGER.info("running %d ablations on %d videos", len(ABLATIONS), len(specs))

    base_config = DetectorConfig()
    base_config.kalman = config.get("front_end", {}).get("kalman", {})
    base_config.hsv = config.get("front_end", {}).get("hsv", {})

    selected = args.ablations or list(ABLATIONS)
    results: list[dict[str, Any]] = []
    for name in selected:
        if name not in ABLATIONS:
            raise KeyError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
        LOGGER.info("ablation: %s", name)
        result = run_ablation(
            name,
            ABLATIONS[name],
            specs,
            spatial_checkpoint=args.spatial_checkpoint,
            temporal_checkpoint=args.model,
            device=args.device,
            policy_overrides=args.policy_overrides,
            refractory_seconds=args.refractory_seconds,
            base_config=base_config,
        )
        results.append(result)
        metrics = result["video_metrics"]
        LOGGER.info(
            "  %-32s recall=%.3f fa/h=%.3f delay=%.2fs",
            name,
            metrics["video_recall"],
            metrics["false_alarms_per_hour"],
            metrics.get("delay_mean", float("nan")),
        )

    name = args.tag or experiment_id("ablations", split=args.split, seed=args.seed)
    payload = {
        "split": args.split,
        "role": args.role,
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "refractory_seconds": args.refractory_seconds,
        "ablations": results,
    }
    write_result(args.out_dir, name, payload)

    csv_path = Path(args.out_dir) / f"{name}_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ablation",
        "description",
        "video_recall",
        "video_specificity",
        "false_alarms_per_hour",
        "delay_mean",
        "delay_p95",
        "recall_at_0_5s",
        "recall_at_1_0s",
        "recall_at_2_0s",
        "n_false_alarm_events",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["video_metrics"]
            writer.writerow(
                {
                    "ablation": result["ablation"],
                    "description": result["description"],
                    **{key: metrics.get(key, "") for key in fields[2:]},
                }
            )
    LOGGER.info("wrote %s", csv_path)

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
        model="ablations",
        notes="ablation grid",
        metrics={"n_ablations": len(results)},
    )
    append_to_registry(args.registry, record)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
