"""Paired statistical comparisons between two evaluated models.

The unit of analysis is the **video**, never the frame: frames of a video are
temporally correlated and would grossly understate the uncertainty.

Three tests are provided:

* a paired bootstrap over the per-video metric difference,
* the Wilcoxon signed-rank test over the per-video metric differences,
* McNemar's test for paired binary outcomes (e.g. "was the video detected?").

Example
-------
    python -m src.evaluation.statistics \\
        --baseline results/raw/video_eval_baseline.json \\
        --candidate results/raw/video_eval_tcn.json \\
        --metric false_alarms_per_hour
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger, setup_logging
from .metrics import mcnemar_test, paired_bootstrap, wilcoxon_signed_rank

__all__ = ["load_per_video", "compare", "main"]

LOGGER = get_logger("evaluation.statistics")


def load_per_video(path: str | Path) -> tuple[dict[str, dict], dict[str, Any]]:
    """Load a video evaluation report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_video = {row["video_id"]: row for row in payload.get("per_video", [])}
    return per_video, payload.get("video_metrics", {})


def _per_video_values(
    per_a: dict[str, dict],
    per_b: dict[str, dict],
    metric: str,
    *,
    lower_is_better: bool | None = None,
) -> tuple[list[str], list[float], list[float]]:
    """Extract the per-video values of a metric for both systems.

    ``false_alarms_per_hour`` is a corpus-level quantity, so it is decomposed
    into its per-video contributions (the number of false alarm events of each
    negative video) before the paired test.
    """
    shared = sorted(set(per_a) & set(per_b))
    a: list[float] = []
    b: list[float] = []
    for video_id in shared:
        row_a = per_a[video_id]
        row_b = per_b[video_id]
        if metric == "false_alarms_per_hour":
            a.append(float(row_a.get("n_false_alarm_events", 0)))
            b.append(float(row_b.get("n_false_alarm_events", 0)))
        elif metric in ("video_recall", "detected"):
            a.append(1.0 if row_a.get("detected") else 0.0)
            b.append(1.0 if row_b.get("detected") else 0.0)
        elif metric == "detection_delay":
            delay_a = row_a.get("detection_delay")
            delay_b = row_b.get("detection_delay")
            if delay_a is None or delay_b is None:
                continue
            a.append(float(delay_a))
            b.append(float(delay_b))
        else:
            a.append(float(row_a.get(metric, 0.0) or 0.0))
            b.append(float(row_b.get(metric, 0.0) or 0.0))
    return shared, a, b


def compare(
    baseline_path: str | Path,
    candidate_path: str | Path,
    metric: str = "detection_delay",
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the paired tests between two evaluated systems."""
    per_a, metrics_a = load_per_video(baseline_path)
    per_b, metrics_b = load_per_video(candidate_path)

    shared, a, b = _per_video_values(per_a, per_b, metric)
    if not shared:
        return {"metric": metric, "error": "no paired videos"}

    result: dict[str, Any] = {
        "metric": metric,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "n_paired_videos": len(shared),
        "baseline_corpus_value": metrics_a.get(metric),
        "candidate_corpus_value": metrics_b.get(metric),
    }

    if metric in ("video_recall", "detected"):
        result["mcnemar"] = mcnemar_test(np.array(a) > 0.5, np.array(b) > 0.5)
    else:
        result["paired_bootstrap"] = paired_bootstrap(b, a, n_resamples=n_resamples, seed=seed)
        result["wilcoxon"] = wilcoxon_signed_rank(b, a)

    result["per_video"] = [
        {"video_id": video_id, "baseline": va, "candidate": vb} for video_id, va, vb in zip(shared, a, b)
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Paired statistical comparison.")
    parser.add_argument("--baseline", required=True, help="baseline video evaluation JSON")
    parser.add_argument("--candidate", required=True, help="candidate video evaluation JSON")
    parser.add_argument("--metric", default="detection_delay")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    result = compare(
        args.baseline,
        args.candidate,
        args.metric,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
