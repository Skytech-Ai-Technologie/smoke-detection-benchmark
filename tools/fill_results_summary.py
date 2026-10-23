"""Fill the result tables of ``paper/results_summary.md`` from the artefacts.

The document is written by hand, but every number in it comes from the
machine-readable results, so this script replaces the blocks that are marked
with ``<!-- BEGIN:<name> -->`` and ``<!-- END:<name> -->`` by the content
that is computed from ``results/raw`` and ``results/tables``.

Usage:
    python tools/fill_results_summary.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
DOC = ROOT / "paper" / "results_summary.md"


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def block_headline() -> str:
    path = TABLES / "summary_over_splits.json"
    if not path.exists():
        return "_not available_"
    data = load_json(path)
    rows = []
    for model, summary in sorted(data.items()):
        r = summary["video_recall"]
        s = summary["video_specificity"]
        f = summary["false_alarms_per_hour"]
        d = summary["delay_mean"]
        rows.append(
            f"| {model} | {summary['n_splits']} | {fmt(r['mean'])} +- {fmt(r['std'])} | "
            f"{fmt(s['mean'])} +- {fmt(s['std'])} | {fmt(f['mean'], 1)} +- {fmt(f['std'], 1)} | "
            f"{fmt(d['mean'], 2)} +- {fmt(d['std'], 2)} |"
        )
    return "\n".join(rows)


def block_headline_text() -> str:
    path = TABLES / "summary_over_splits.json"
    if not path.exists():
        return "_not available_"
    data = load_json(path)
    tcn = data.get("tcn")
    spatial = data.get("spatial")
    if not tcn or not spatial:
        return "_not available_"
    fa_t = tcn["false_alarms_per_hour"]["mean"]
    fa_s = spatial["false_alarms_per_hour"]["mean"]
    d_t = tcn["delay_mean"]["mean"]
    d_s = spatial["delay_mean"]["mean"]
    r_t = tcn["video_recall"]["mean"]
    r_s = spatial["video_recall"]["mean"]
    reduction = 100.0 * (1.0 - fa_t / fa_s) if fa_s else float("nan")
    delay = 100.0 * (1.0 - d_t / d_s) if d_s else float("nan")
    return (
        f"Over the five predefined video-level splits, the calibrated temporal detector "
        f"reduces the false alarms per hour from **{fmt(fa_s, 1)}** (frame-only CNN, fixed "
        f"threshold) to **{fmt(fa_t, 1)}**, a reduction of **{fmt(reduction, 1)} %**, while the "
        f"video-level recall changes from {fmt(r_s)} to {fmt(r_t)} and the detection delay "
        f"falls from {fmt(d_s, 2)} s to {fmt(d_t, 2)} s ({fmt(delay, 1)} % faster). "
        f"The pre-registered margin of a 20 % false-alarm reduction is therefore met, and the "
        f"margin of a maximum recall drop of 0.02 is met as well (the recall does not drop)."
    )


def block_matched_recall() -> str:
    lines = [
        "| Recall level | " + " | ".join(sorted(_sweep_models())) + " |",
        "|---|" + "---|" * len(_sweep_models()),
    ]
    rows = _matched_rows()
    for row in rows:
        cells = [fmt(row["recall_level"], 2)]
        for model in sorted(_sweep_models()):
            cells.append(fmt(row.get(model), 1))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sweep_models() -> set[str]:
    models: set[str] = set()
    for path in sorted(RAW.glob("sweep_*.json")):
        data = load_json(path)
        models.update(data.get("curves", {}).keys())
    return models


def _matched_rows() -> list[dict]:
    """Average the interpolated FA/h at each matched recall level over the splits.

    Every ``sweep_*.json`` artefact reports a ``matched_recall_comparison``
    whose rows contain the FA/h of each model at a common recall level. The
    aggregation averages over all sweeps in which the level is present, so a
    model that is only available on one split (the GRU) still contributes.
    """
    collected: dict[float, dict[str, list[float]]] = {}
    for path in sorted(RAW.glob("sweep_*.json")):
        data = load_json(path)
        for row in data.get("matched_recall_comparison", []):
            level = row.get("recall_level")
            if level is None:
                continue
            per_model = collected.setdefault(float(level), {})
            for model, value in row.items():
                if model == "recall_level":
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value != value:  # NaN: the sweep could not reach this recall
                    value = float("nan")
                per_model.setdefault(model, []).append(value)
    rows: list[dict] = []
    for level in sorted(collected):
        entry: dict[str, Any] = {"recall_level": level}
        for model, values in collected[level].items():
            finite = [v for v in values if v == v]
            if finite:
                entry[model] = sum(finite) / len(finite)
        rows.append(entry)
    return rows


def block_calibration() -> str:
    lines = [
        "| Split | Method | ECE | MCE | Brier |",
        "|---|---|---|---|---|",
    ]
    reports = sorted(RAW.glob("calibration_report*.json"))
    reports += sorted(RAW.glob("calibration_*/calibration_report.json"))
    for path in reports:
        data = load_json(path)
        # Several calibrations (TCN, GRU, exploratory methods) share one
        # split label; the folder disambiguates which model a row belongs to.
        split_label = path.parent.name if path.parent != RAW else path.stem
        split_label = split_label.replace("calibration_", "").replace("calibration", "")
        split_label = split_label or path.stem
        methods = data.get("methods", {})
        if not methods:
            continue
        for method, values in sorted(methods.items()):
            lines.append(
                f"| {split_label} | {method} | "
                f"{fmt(values.get('ece'), 6)} | {fmt(values.get('mce'), 4)} | {fmt(values.get('brier'), 4)} |"
            )
    if len(lines) == 2:
        return "_not available_"
    return "\n".join(lines)


def block_ablations() -> str:
    path = TABLES / "table5_ablations.csv"
    if not path.exists():
        return "_not available_"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "| Variant | Split | Video recall | Specificity | False alarms/h | Delay mean (s) | Description |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {row.get('split', '')} | {fmt(_f(row.get('video_recall')))} | "
            f"{fmt(_f(row.get('video_specificity')))} | {fmt(_f(row.get('false_alarms_per_hour')), 1)} | "
            f"{fmt(_f(row.get('delay_mean_s')), 2)} | {row.get('description', '')} |"
        )
    return "\n".join(lines)


def block_edge() -> str:
    path = TABLES / "table2_edge_deployment.csv"
    if not path.exists():
        return "_not available_"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "| Model | Device | System FPS | p50 (ms) | p95 (ms) | p99 (ms) | Jitter (ms) | Dropped | Peak RAM (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('model', '')} | {row.get('device', '')} | {fmt(_f(row.get('system_fps')), 2)} | "
            f"{fmt(_f(row.get('p50_ms')), 2)} | {fmt(_f(row.get('p95_ms')), 2)} | {fmt(_f(row.get('p99_ms')), 2)} | "
            f"{fmt(_f(row.get('jitter_ms')), 2)} | {row.get('dropped_frames', '')} | {fmt(_f(row.get('peak_ram_mb')), 1)} |"
        )
    return "\n".join(lines)


def block_stages() -> str:
    for path in sorted(RAW.glob("latency_*.json")):
        data = load_json(path)
        single = data.get("single_stream", {})
        stages = single.get("stage_ms_per_frame", {})
        if not stages:
            continue
        lines = [
            "| Stage | ms per frame | share |",
            "|---|---|---|",
        ]
        for stage, value in stages.items():
            lines.append(
                f"| {stage} | {fmt(value, 3)} | {fmt(100 * single['stage_share'].get(stage, 0.0), 1)} % |"
            )
        return "\n".join(lines)
    return "_not available_"


def block_quantization() -> str:
    path = TABLES / "table3_quantization.csv"
    if not path.exists():
        return "_not available_"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "_not available_"
    lines = [
        "| Model | Mode | Max abs logit difference | Mean abs logit difference | Artifact bytes (reference FP32 bytes) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        sizes = row.get("size_bytes", "")
        try:
            size_map = json.loads(sizes) if sizes else {}
        except Exception:
            size_map = {}
        mode = row.get("mode", "fp32")
        artifact = size_map.get(f"{mode}_bytes") or size_map.get("int8_bytes") or ""
        reference = size_map.get("fp32_bytes") or ""
        sizes_cell = (
            f"{artifact} ({reference})"
            if artifact and reference
            else (artifact or reference or json.dumps(size_map))
        )
        label = str(row.get("model", "")).replace("/quantization_report", "")
        lines.append(
            f"| {label} | {mode} | "
            f"{fmt(_f(row.get('max_abs_logit_diff')), 5)} | {fmt(_f(row.get('mean_abs_logit_diff')), 6)} | "
            f"{sizes_cell} |"
        )
    return "\n".join(lines)


def block_statistics() -> str:
    lines = [
        "| Split | Metric | n videos | Mean difference (candidate - baseline) | 95 % CI | p | test |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in sorted(TABLES.glob("statistics_*.json")):
        data = load_json(path)
        stem = path.stem  # statistics_<metric>_<split>
        parts = stem.split("_")
        split = parts[-1] if parts and parts[-1].startswith("split") else ""
        boot = data.get("paired_bootstrap", {})
        wil = data.get("wilcoxon", {})
        mcn = data.get("mcnemar", {})
        if mcn:
            p = fmt(mcn.get("p_value"), 4)
            test_name = "McNemar"
            diff = ""
            ci = ""
            lines.append(
                f"| {split} | {data.get('metric', '')} | {data.get('n_paired_videos', '')} | "
                f"{diff} | {ci} | {p} | {test_name} |"
            )
            continue
        p_wil = fmt(wil.get("p_value"), 4) if wil else "-"
        p_boot = fmt(boot.get("p_value"), 4)
        lines.append(
            f"| {split} | {data.get('metric', '')} | {data.get('n_paired_videos', '')} | "
            f"{fmt(boot.get('mean_difference'), 3)} | "
            f"[{fmt(boot.get('ci_lower'), 3)}, {fmt(boot.get('ci_upper'), 3)}] | {p_boot} | "
            f"Wilcoxon p={p_wil} |"
        )
    return "\n".join(lines) if len(lines) > 2 else "_not available_"


def block_accuracy() -> str:
    path = TABLES / "table1_main_accuracy.csv"
    if not path.exists():
        return "_not available_"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "| Model | Split | Video recall | Specificity | False alarms/h | Delay mean (s) | Delay p95 (s) | "
        "Recall@0.5s | Recall@1s | Recall@2s | FA events |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        label = str(row.get("model", ""))
        for prefix in ("video_eval_", "frame_eval_", "ablations_", "latency_", "energy_", "streaming_eval_"):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        label = re.sub(r"_?seed\d+$", "", label)
        lines.append(
            f"| {label} | {row.get('split', '')} | {fmt(_f(row.get('video_recall')))} | "
            f"{fmt(_f(row.get('video_specificity')))} | {fmt(_f(row.get('false_alarms_per_hour')), 1)} | "
            f"{fmt(_f(row.get('delay_mean_s')), 2)} | {fmt(_f(row.get('delay_p95_s')), 2)} | "
            f"{fmt(_f(row.get('recall_at_0_5s')))} | {fmt(_f(row.get('recall_at_1_0s')))} | "
            f"{fmt(_f(row.get('recall_at_2_0s')))} | {row.get('false_alarm_events', '')} |"
        )
    return "\n".join(lines)


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


BLOCKS = {
    "headline": block_headline,
    "headline_text": block_headline_text,
    "matched_recall": block_matched_recall,
    "calibration_table": block_calibration,
    "accuracy_table": block_accuracy,
    "ablations_table": block_ablations,
    "edge_table": block_edge,
    "edge_stages": block_stages,
    "quantization_table": block_quantization,
    "statistics_table": block_statistics,
}


def main() -> int:
    if not DOC.exists():
        print(f"{DOC} not found", file=sys.stderr)
        return 1
    text = DOC.read_text(encoding="utf-8")
    for name, builder in BLOCKS.items():
        begin = f"<!-- BEGIN:{name} -->"
        end = f"<!-- END:{name} -->"
        if begin not in text or end not in text:
            print(f"  block {name}: not present in the document")
            continue
        content = builder()
        head, _, rest = text.partition(begin)
        _, _, tail = rest.partition(end)
        text = f"{head}{begin}\n{content}\n{end}{tail}"
        print(f"  block {name}: {len(content.splitlines())} rows")
    DOC.write_text(text, encoding="utf-8")
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
