"""Build the paper-ready result tables.

Reads the machine-readable reports in ``results/raw`` and writes the five
tables required by the specification into ``results/tables``:

1. main accuracy table,
2. edge deployment table,
3. quantisation table,
4. generalisation table,
5. ablation table.

Example
-------
    python -m src.evaluation.make_tables --results results/raw --out results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger, setup_logging

__all__ = ["main"]

LOGGER = get_logger("evaluation.make_tables")


def _load_reports(results_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    reports: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            # Reports may live at the top level or inside per-experiment
            # subdirectories. The relative path keeps every entry unique and
            # still matches the top-level name for flat files.
            name = str(path.relative_to(results_dir).with_suffix("")).replace("\\", "/")
            reports.append((name, payload))
    return reports


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    LOGGER.info("wrote %s (%d rows)", path, len(rows))
    return path


def _fmt(value: Any, digits: int = 4) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return round(value, digits)
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the paper tables.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    reports = _load_reports(results_dir)

    # ---------------------------------------------------------- main accuracy
    main_rows: list[dict[str, Any]] = []
    for name, payload in reports:
        metrics = payload.get("video_metrics")
        if not metrics:
            continue
        main_rows.append(
            {
                "model": name,
                "calibration": payload.get("calibration", "n/a"),
                "split": payload.get("split", ""),
                "video_recall": _fmt(metrics.get("video_recall")),
                "video_specificity": _fmt(metrics.get("video_specificity")),
                "false_alarms_per_hour": _fmt(metrics.get("false_alarms_per_hour"), 3),
                "delay_mean_s": _fmt(metrics.get("delay_mean"), 3),
                "delay_p95_s": _fmt(metrics.get("delay_p95"), 3),
                "recall_at_0_5s": _fmt(metrics.get("recall_at_0_5s")),
                "recall_at_1_0s": _fmt(metrics.get("recall_at_1_0s")),
                "recall_at_2_0s": _fmt(metrics.get("recall_at_2_0s")),
                "n_positive_videos": metrics.get("n_positive_videos", ""),
                "n_negative_videos": metrics.get("n_negative_videos", ""),
                "false_alarm_events": metrics.get("false_alarm_events", ""),
            }
        )
    _write_csv(
        out_dir / "table1_main_accuracy.csv",
        main_rows,
        [
            "model",
            "calibration",
            "split",
            "video_recall",
            "video_specificity",
            "false_alarms_per_hour",
            "delay_mean_s",
            "delay_p95_s",
            "recall_at_0_5s",
            "recall_at_1_0s",
            "recall_at_2_0s",
            "n_positive_videos",
            "n_negative_videos",
            "false_alarm_events",
        ],
    )

    # ------------------------------------------------------ edge deployment
    edge_rows: list[dict[str, Any]] = []
    for name, payload in reports:
        single = payload.get("single_stream")
        if not single:
            continue
        edge_rows.append(
            {
                "model": name,
                "device": payload.get("device", ""),
                "system_fps": _fmt(single.get("system_fps"), 2),
                "p50_ms": _fmt(single.get("p50_ms"), 3),
                "p95_ms": _fmt(single.get("p95_ms"), 3),
                "p99_ms": _fmt(single.get("p99_ms"), 3),
                "jitter_ms": _fmt(single.get("jitter_ms"), 3),
                "dropped_frames": single.get("dropped_frames", ""),
                "peak_ram_mb": _fmt(single.get("peak_ram_mb"), 1),
                "peak_vram_mb": _fmt(single.get("peak_vram_mb"), 1),
                "stages": json.dumps(single.get("stage_ms_per_frame", {})),
            }
        )
    _write_csv(
        out_dir / "table2_edge_deployment.csv",
        edge_rows,
        [
            "model",
            "device",
            "system_fps",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "jitter_ms",
            "dropped_frames",
            "peak_ram_mb",
            "peak_vram_mb",
            "stages",
        ],
    )

    # -------------------------------------------------------- quantisation
    quant_rows: list[dict[str, Any]] = []
    for name, payload in reports:
        if "comparison" not in payload or "sizes" not in payload:
            continue
        quant_rows.append(
            {
                "model": name,
                "mode": payload.get("mode", ""),
                "source_checkpoint": payload.get("source_checkpoint", ""),
                "max_abs_logit_diff": _fmt(payload["comparison"].get("max_abs_diff"), 5),
                "mean_abs_logit_diff": _fmt(payload["comparison"].get("mean_abs_diff"), 6),
                "size_bytes": json.dumps(payload.get("sizes", {})),
            }
        )
    _write_csv(
        out_dir / "table3_quantization.csv",
        quant_rows,
        [
            "model",
            "mode",
            "source_checkpoint",
            "max_abs_logit_diff",
            "mean_abs_logit_diff",
            "size_bytes",
        ],
    )

    # ------------------------------------------------------ generalisation
    gen_rows: list[dict[str, Any]] = []
    for name, payload in reports:
        if "external_dataset" not in payload:
            continue
        metrics = payload.get("video_metrics", {})
        gen_rows.append(
            {
                "train_dataset": payload.get("train_dataset", ""),
                "test_dataset": payload.get("external_dataset", ""),
                "model": name,
                "video_recall": _fmt(metrics.get("video_recall")),
                "video_specificity": _fmt(metrics.get("video_specificity")),
                "false_alarms_per_hour": _fmt(metrics.get("false_alarms_per_hour"), 3),
                "delay_mean_s": _fmt(metrics.get("delay_mean"), 3),
            }
        )
    _write_csv(
        out_dir / "table4_generalization.csv",
        gen_rows,
        [
            "train_dataset",
            "test_dataset",
            "model",
            "video_recall",
            "video_specificity",
            "false_alarms_per_hour",
            "delay_mean_s",
        ],
    )

    # ------------------------------------------------------------ ablations
    ablation_rows: list[dict[str, Any]] = []
    for name, payload in reports:
        for ablation in payload.get("ablations", []):
            metrics = ablation.get("video_metrics", {})
            ablation_rows.append(
                {
                    "variant": ablation.get("ablation", ""),
                    "description": ablation.get("description", ""),
                    "split": payload.get("split", ""),
                    "video_recall": _fmt(metrics.get("video_recall")),
                    "video_specificity": _fmt(metrics.get("video_specificity")),
                    "false_alarms_per_hour": _fmt(metrics.get("false_alarms_per_hour"), 3),
                    "delay_mean_s": _fmt(metrics.get("delay_mean"), 3),
                    "delay_p95_s": _fmt(metrics.get("delay_p95"), 3),
                    "n_false_alarm_events": metrics.get("false_alarm_events", ""),
                }
            )
    _write_csv(
        out_dir / "table5_ablations.csv",
        ablation_rows,
        [
            "variant",
            "description",
            "split",
            "video_recall",
            "video_specificity",
            "false_alarms_per_hour",
            "delay_mean_s",
            "delay_p95_s",
            "n_false_alarm_events",
        ],
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
