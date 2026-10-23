"""Select the alarm operating point on the validation split.

Every model in the benchmark is evaluated at the operating point that is
*optimal for it on the validation videos*, and that operating point is then
applied unchanged to the test split. This is the only fair way to compare a
fixed threshold rule with a calibrated temporal model: the same rule is
allowed to use the threshold and the persistence duration that suits it best,
but it may never be tuned on the test data.

The selection objective is the one registered in ``paper/methods_notes.md``:

    maximise   video_recall
    subject to false_alarms_per_hour <= max_false_alarms_per_hour

If no point satisfies the budget, the point that maximises
``video_recall - penalty * false_alarms_per_hour`` is used instead.

The sweep also produces the *false alarms per hour versus recall* curve that
is reported as a figure in the paper.

Example
-------
    python -m src.training.select_thresholds \\
        --model results/raw/tcn_split01_seed42/model_calibrated.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --split split01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models.calibrated_detector import AlarmPolicy, AlarmPolicyConfig
from ..models.builder import build_detector
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from ..evaluation.metrics import aggregate_video_metrics, evaluate_video_events
from ..evaluation.runner import load_specs_from_manifest

__all__ = ["sweep_operating_points", "select_operating_point", "main"]

LOGGER = get_logger("training.select_thresholds")

DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
DEFAULT_PERSISTENCE = (0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0)


def _replay_policy(
    probabilities: np.ndarray,
    uncertainties: np.ndarray,
    reliabilities: np.ndarray,
    *,
    threshold: float,
    persistence_seconds: float,
    fps: float,
    refractory_seconds: float,
    hysteresis_ratio: float,
    mode: str,
    max_persistence_seconds: float,
) -> np.ndarray:
    """Replay the alarm policy over a cached probability stream."""
    policy = AlarmPolicy(
        AlarmPolicyConfig(
            mode=mode,
            confidence_threshold=threshold,
            persistence_seconds=persistence_seconds,
            refractory_seconds=refractory_seconds,
            hysteresis_ratio=hysteresis_ratio,
            max_persistence_seconds=max_persistence_seconds,
        ),
        fps=fps,
    )
    mask = np.zeros(len(probabilities), dtype=bool)
    for t, probability in enumerate(probabilities):
        out = policy.step(
            float(probability),
            frame_index=t,
            timestamp=t / fps,
            uncertainty=float(uncertainties[t]),
            motion_reliability=float(reliabilities[t]),
        )
        mask[t] = bool(out["alarm"])
    policy.close(len(probabilities), len(probabilities) / fps)
    return mask


def sweep_operating_points(
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
    max_persistence_seconds: float = 3.0,
    max_false_alarms_per_hour: float = 1.0,
    false_alarm_penalty: float = 1.0,
    variant: str | None = None,
) -> dict[str, Any]:
    """Sweep the alarm operating point on the given (validation) videos."""
    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides={},
        fps=25.0,
        variant=variant,
    )
    detector = bundle.detector

    # 1) Run the front end and the temporal model once per video and cache the
    #    per-frame quantities that the policy depends on.
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
    LOGGER.info("cached %d validation streams", len(streams))

    # 2) Sweep the policy over the cached streams.
    curve: list[dict[str, Any]] = []
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
                    max_persistence_seconds=max_persistence_seconds,
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
            curve.append(
                {
                    "threshold": float(threshold),
                    "persistence_seconds": float(persistence),
                    "mode": mode,
                    "video_recall": metrics["video_recall"],
                    "video_specificity": metrics["video_specificity"],
                    "false_alarms_per_hour": metrics["false_alarms_per_hour"],
                    "false_alarm_events": metrics["false_alarm_events"],
                    "delay_mean": metrics.get("delay_mean", float("nan")),
                    "delay_p95": metrics.get("delay_p95", float("nan")),
                    "recall_at_0_5s": metrics["recall_at_0_5s"],
                    "recall_at_1_0s": metrics["recall_at_1_0s"],
                    "recall_at_2_0s": metrics["recall_at_2_0s"],
                }
            )
            LOGGER.info(
                "threshold=%.2f persistence=%.1fs -> recall=%.3f fa/h=%.2f",
                threshold,
                persistence,
                curve[-1]["video_recall"],
                curve[-1]["false_alarms_per_hour"],
            )

    selection = select_operating_point(
        curve,
        max_false_alarms_per_hour=max_false_alarms_per_hour,
        false_alarm_penalty=false_alarm_penalty,
    )
    return {"curve": curve, "selection": selection}


def select_operating_point(
    curve: Sequence[dict[str, Any]],
    *,
    max_false_alarms_per_hour: float = 1.0,
    false_alarm_penalty: float = 1.0,
) -> dict[str, Any]:
    """Pick the operating point on the validation curve.

    The rule is: maximise the video recall subject to a false-alarm budget;
    if the budget cannot be met, maximise
    ``video_recall - penalty * false_alarms_per_hour``.
    """
    if not curve:
        return {}

    feasible = [p for p in curve if p["false_alarms_per_hour"] <= max_false_alarms_per_hour]
    if feasible:
        best = max(
            feasible,
            key=lambda p: (p["video_recall"], -p["false_alarms_per_hour"], -p["persistence_seconds"]),
        )
        reason = "maximise recall subject to the false-alarm budget"
    else:
        best = max(
            curve,
            key=lambda p: (
                p["video_recall"] - false_alarm_penalty * p["false_alarms_per_hour"],
                -p["false_alarms_per_hour"],
            ),
        )
        reason = "no point met the false-alarm budget; maximise recall - penalty * fa/h"

    return {
        **best,
        "selection_rule": reason,
        "max_false_alarms_per_hour": max_false_alarms_per_hour,
        "false_alarm_penalty": false_alarm_penalty,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Select the alarm operating point.")
    parser.add_argument("--model", required=True, help="temporal checkpoint (or 'none')")
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--split", default="split01")
    parser.add_argument("--role", default="validation")
    parser.add_argument("--manifest", default="data/manifests/firesense_videos.csv")
    parser.add_argument("--feature-dir", default="data/processed/features")
    parser.add_argument("--splits-file", default="data/manifests/splits.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument(
        "--variant",
        default=None,
        help="reference model variant: spatial, voting_mean, voting_median, "
        "voting_majority, voting_max or voting_persistence",
    )
    parser.add_argument("--max-false-alarms-per-hour", type=float, default=1.0)
    parser.add_argument("--false-alarm-penalty", type=float, default=1.0)
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
    LOGGER.info("selecting the operating point on %d %s videos", len(specs), args.role)

    result = sweep_operating_points(
        specs,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=device,
        mode=args.mode,
        max_false_alarms_per_hour=args.max_false_alarms_per_hour,
        false_alarm_penalty=args.false_alarm_penalty,
        variant=args.variant,
    )

    name = args.tag or experiment_id(f"thresholds_{args.mode}", split=args.split, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "split": args.split,
        "role": args.role,
        "device": device,
        "curve": result["curve"],
        "selection": result["selection"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", path)

    record = build_record(
        exp_id=name,
        config={"mode": args.mode, "split": args.split, "role": args.role},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        split=args.split,
        model="threshold_selection",
        notes="operating-point selection on the validation split",
        metrics=result["selection"],
    )
    append_to_registry(args.registry, record)

    LOGGER.info("selected operating point: %s", json.dumps(result["selection"], indent=2))
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
