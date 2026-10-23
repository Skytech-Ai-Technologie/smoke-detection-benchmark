"""Online streaming evaluation.

The streaming mode is the deployment mode: the frames are read one by one
from the source and the detector is updated after every frame, exactly as it
would be on a live camera. The detector never sees a future frame, so the
result must be identical to the offline video evaluation.

This script therefore does two things:

1. it evaluates a video in streaming mode and reports the same video-level
   metrics as the offline evaluation,
2. it compares the two and reports the **agreement** between them, which is
   the check that the offline and the online paths cannot silently disagree.

Example
-------
    python -m src.evaluation.evaluate_streaming \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn_split01_seed42/model_calibrated.pt \\
        --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt \\
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
from .runner import load_specs_from_manifest

__all__ = ["main"]

LOGGER = get_logger("evaluation.evaluate_streaming")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Online streaming evaluation.")
    add_common_arguments(parser)
    parser.add_argument("--reference", default=None, help="offline evaluation JSON to compare with")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    resolve_config(args)

    import cv2

    from ..models.builder import build_detector

    specs = load_specs_from_manifest(
        args.manifest,
        args.feature_dir,
        role=args.role,
        split=args.split,
        splits_file=args.splits_file,
    )
    if not specs:
        raise RuntimeError("no videos selected for the streaming evaluation")

    bundle = build_detector(
        args.spatial_checkpoint,
        args.model,
        device=args.device,
        policy_overrides=args.policy_overrides,
    )
    detector = bundle.detector

    evaluations = []
    per_video: list[dict[str, Any]] = []
    for spec in specs:
        detector.reset(spec.video_id)
        capture = cv2.VideoCapture(spec.path)
        if not capture.isOpened():
            raise RuntimeError(f"could not open {spec.path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        mask: list[bool] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # A live stream: the decision for frame t is taken before frame
            # t+1 is decoded.
            record = detector.process_frame(frame, index, timestamp=index / fps)
            mask.append(record.alarm_state == "alarm")
            index += 1
        capture.release()
        detector.policy.close(index, index / fps)

        evaluation = evaluate_video_events(
            spec.video_id,
            spec.label,
            fps,
            np.asarray(mask, dtype=bool),
            spec.onset_frame,
            refractory_seconds=args.refractory_seconds,
        )
        evaluations.append(evaluation)
        per_video.append(evaluation.to_dict())

    metrics = aggregate_video_metrics(evaluations, refractory_seconds=args.refractory_seconds)

    agreement: dict[str, Any] = {}
    if args.reference:
        import json

        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        ref_rows = {row["video_id"]: row for row in reference.get("per_video", [])}
        identical = 0
        compared = 0
        for row in per_video:
            ref = ref_rows.get(row["video_id"])
            if ref is None:
                continue
            compared += 1
            if (
                ref["detected"] == row["detected"]
                and ref["n_alarm_events"] == row["n_alarm_events"]
                and ref["n_false_alarm_events"] == row["n_false_alarm_events"]
            ):
                identical += 1
        agreement = {
            "reference": args.reference,
            "n_videos_compared": compared,
            "n_identical": identical,
            "agreement": identical / compared if compared else float("nan"),
        }
        LOGGER.info(
            "streaming/offline agreement: %d/%d videos identical",
            identical,
            compared,
        )

    payload = {
        "split": args.split,
        "role": args.role,
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "mode": "streaming",
        "video_metrics": metrics,
        "per_video": per_video,
        "agreement_with_offline": agreement,
    }

    name = args.tag or experiment_id("streaming_eval", split=args.split, seed=args.seed)
    write_result(args.out_dir, name, payload)

    csv_path = Path(args.out_dir) / f"{name}_per_video.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        for row in per_video:
            writer.writerow(row)

    record = build_record(
        exp_id=name,
        config={"split": args.split, "role": args.role, "mode": "streaming"},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        split=args.split,
        model="streaming_eval",
        notes="online streaming evaluation",
        metrics=metrics,
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "streaming video metrics | recall=%.3f specificity=%.3f fa/h=%.2f delay=%.2fs",
        metrics["video_recall"],
        metrics["video_specificity"],
        metrics["false_alarms_per_hour"],
        metrics.get("delay_mean", float("nan")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
