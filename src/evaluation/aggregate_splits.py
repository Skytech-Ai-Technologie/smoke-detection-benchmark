"""Aggregate the results of the five predefined splits.

Because FIRESENSE contains only 22 videos, a single split is not an
estimate of anything. This module aggregates the per-split results into the
mean and the standard deviation that the paper reports, and it also reports
the per-video outcomes, which are the unit of analysis of every statistical
test.

Example
-------
    python -m src.evaluation.aggregate_splits --results results/raw
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger, setup_logging

__all__ = ["collect_runs", "aggregate", "main"]

LOGGER = get_logger("evaluation.aggregate_splits")

METRICS = [
    "video_recall",
    "video_specificity",
    "false_alarms_per_hour",
    "false_alarm_events",
    "delay_mean",
    "delay_p95",
    "recall_at_0_5s",
    "recall_at_1_0s",
    "recall_at_2_0s",
]


def collect_runs(results_dir: Path, prefix: str) -> dict[str, list[dict[str, Any]]]:
    """Group the runs of a given kind by the model name they belong to."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # The file names are "<prefix>_<model>_<split>_seed<seed>.json"; the model
    # part is optional (the primary model of a split has no name in it).
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(?:(?P<model>.+?)_)?(?P<split>split\d+)_seed(?P<seed>\d+)"
    )
    for path in sorted(results_dir.glob(f"{prefix}_*.json")):
        match = pattern.match(path.stem)
        if not match:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = payload.get("video_metrics")
        if not metrics:
            continue
        grouped[match.group("model") or prefix].append(
            {
                "split": match.group("split"),
                "seed": int(match.group("seed")),
                "metrics": metrics,
                "per_video": payload.get("per_video", []),
                "path": str(path),
            }
        )
    return grouped


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of every metric over the splits."""
    out: dict[str, Any] = {"n_splits": len(runs), "splits": [r["split"] for r in runs]}
    for metric in METRICS:
        values = [float(r["metrics"].get(metric, float("nan"))) for r in runs]
        values = [v for v in values if v == v]
        if not values:
            out[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        out[metric] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Aggregate the results over the splits.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--prefix", default="video_eval")
    parser.add_argument("--out", default="results/tables/summary_over_splits.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    results_dir = Path(args.results)
    grouped = collect_runs(results_dir, args.prefix)

    rows: list[dict[str, Any]] = []
    for model, runs in sorted(grouped.items()):
        summary = aggregate(runs)
        row: dict[str, Any] = {"model": model, "n_splits": summary["n_splits"]}
        for metric in METRICS:
            values = summary[metric]
            row[f"{metric}_mean"] = round(values["mean"], 4) if values["n"] else ""
            row[f"{metric}_std"] = round(values["std"], 4) if values["n"] else ""
        rows.append(row)
        LOGGER.info(
            "%-28s recall=%.3f+-%.3f fa/h=%.2f+-%.2f delay=%.2f+-%.2f",
            model,
            summary["video_recall"]["mean"],
            summary["video_recall"]["std"],
            summary["false_alarms_per_hour"]["mean"],
            summary["false_alarms_per_hour"]["std"],
            summary["delay_mean"]["mean"],
            summary["delay_mean"]["std"],
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        LOGGER.info("wrote %s", out_path)

    payload = {model: aggregate(runs) for model, runs in grouped.items()}
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", json_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
