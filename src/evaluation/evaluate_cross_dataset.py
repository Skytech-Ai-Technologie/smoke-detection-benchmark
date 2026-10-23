"""Cross-dataset (external) evaluation.

Measures how well a model trained on FIRESENSE transfers to an independent
video dataset. Only datasets whose files and licences have been verified
individually may be used; the manifest of an external dataset must record the
source URL and the licence.

The external videos are evaluated with the *same* thresholds and the *same*
alarm policy that were selected on the FIRESENSE validation split. No
threshold is re-tuned on the external data, because that would measure
adaptation rather than generalisation.

Example
-------
    python -m src.evaluation.evaluate_cross_dataset \\
        --model results/raw/tcn/model.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --dataset visifire
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..data.build_manifest import build_manifest
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .common import add_common_arguments, resolve_config, write_result
from .metrics import aggregate_video_metrics, evaluate_video_events
from .runner import load_specs_from_manifest, run_detector

__all__ = ["EXTERNAL_DATASETS", "main"]

LOGGER = get_logger("evaluation.evaluate_cross_dataset")

# External datasets that may be used for the generalisation table. Each entry
# documents where the data comes from and under which licence it is used.
EXTERNAL_DATASETS: dict[str, dict[str, Any]] = {
    "visifire": {
        "manifest": "data/manifests/visifire_videos.csv",
        "root": "data/raw/visifire",
        "source_url": "https://github.com/umautobots/visifire",
        "license": "CC BY-NC-SA 4.0",
        "status": "optional",
    },
    "bowfire": {
        "manifest": "data/manifests/bowfire_images.csv",
        "root": "data/raw/bowfire",
        "source_url": "https://github.com/BoWFire",
        "license": "verify before use",
        "status": "spatial sanity check only",
    },
    "dfire": {
        "manifest": "data/manifests/dfire_images.csv",
        "root": "data/raw/dfire",
        "source_url": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "license": "see the official repository",
        "status": "spatial only, never a temporal benchmark",
    },
}


def _ensure_manifest(dataset: str, root: str | None, manifest: str | None) -> str:
    """Return the path of the manifest of an external dataset, building it if needed."""
    entry = EXTERNAL_DATASETS.get(dataset, {})
    manifest_path = manifest or entry.get("manifest", f"data/manifests/{dataset}_videos.csv")
    if Path(manifest_path).exists():
        return manifest_path

    data_root = root or entry.get("root")
    if not data_root or not Path(data_root).exists():
        raise FileNotFoundError(
            f"no manifest for the external dataset {dataset!r} and the data root {data_root!r} "
            "does not exist. Build the manifest first with: "
            f"python -m src.data.build_manifest --dataset {dataset} --root {data_root}"
        )

    build_manifest(
        data_root,
        dataset,
        out_path=manifest_path,
        source_url=entry.get("source_url", ""),
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation.")
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="external dataset name")
    parser.add_argument("--external-root", default=None, help="root directory of the external data")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    resolve_config(args)

    manifest_path = _ensure_manifest(args.dataset, args.external_root, args.manifest)
    LOGGER.info("external dataset %s -> %s", args.dataset, manifest_path)

    # The external set is evaluated as a single "test" block: the split
    # registry of the training dataset does not apply here.
    specs = load_specs_from_manifest(
        manifest_path,
        args.feature_dir,
        role=None,
        split=None,
        splits_file=None,
    )
    if not specs:
        raise RuntimeError(f"no videos found for the external dataset {args.dataset}")
    LOGGER.info("evaluating %d external videos", len(specs))

    results = run_detector(
        specs,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        policy_overrides=args.policy_overrides,
        score_source=args.score_source,
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
        "external_dataset": args.dataset,
        "train_dataset": "firesense",
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "policy": args.policy_overrides,
        "video_metrics": metrics,
        "per_video": per_video,
    }

    name = args.tag or experiment_id(f"cross_{args.dataset}", seed=args.seed)
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
        config={"external_dataset": args.dataset},
        manifest_path=manifest_path,
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        model="cross_dataset",
        notes=f"cross-dataset evaluation on {args.dataset}",
        metrics=metrics,
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "cross-dataset %s | recall=%.3f specificity=%.3f fa/h=%.3f delay=%.2fs",
        args.dataset,
        metrics["video_recall"],
        metrics["video_specificity"],
        metrics["false_alarms_per_hour"],
        metrics.get("delay_mean", float("nan")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
