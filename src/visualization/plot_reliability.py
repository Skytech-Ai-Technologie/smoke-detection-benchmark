"""Reliability diagrams of the *post-calibration* test evaluations.

One panel per evaluated (model, split) pair, so the diagrams can be compared
without overplotting six zig-zag lines into one axes. Each panel shows the
observed positive frequency per score bin as a bar, the per-bin sample count
(bins with no samples are omitted and never drawn), the confusion probability
swept bin-center and the perfect-calibration diagonal.

The scores here were produced by the *calibrated* detector on the test
videos; the calibration fit itself was done on the validation split, so any
misfit visible here is honest generalization error.

Example
-------
    python -m src.visualization.plot_reliability --results results/raw --out results/figures/reliability.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger, setup_logging
from .labels import parse_eval_artifact

__all__ = ["collect_reports", "plot_reliability", "main"]

LOGGER = get_logger("visualization.plot_reliability")


def collect_reports(results_dir: str | Path) -> dict[str, dict]:
    """Load every ``calibration_eval_*.json`` report keyed by identity."""
    reports: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("calibration_eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        identity = parse_eval_artifact(path.stem)
        if identity is None:
            LOGGER.warning("skipping %s (unparsable identity)", path.name)
            continue
        bins = payload.get("reliability_bins")
        if not bins or not any(b.get("count", 0) > 0 for b in bins):
            continue
        key = f"{identity['model']}@{identity['split']}"
        if key not in reports or identity["legacy"] is False:
            reports[key] = {**identity, "payload": payload}
    return dict(sorted(reports.items()))


def plot_reliability(results_dir: str | Path, out_path: str | Path) -> Path:
    """Render one small-multiple reliability panel per evaluated model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = collect_reports(results_dir)
    if not reports:
        raise RuntimeError("no calibration-evaluation reports with reliability bins found")

    n = len(reports)
    n_cols = min(3, max(1, n))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.4 * n_rows), squeeze=False)

    for index, (key, entry) in enumerate(reports.items()):
        ax = axes[index // n_cols][index % n_cols]
        bins = [b for b in entry["payload"].get("reliability_bins", []) if b.get("count", 0) > 0]
        decimals = [b["lower"] for b in bins] + [bins[-1]["upper"]]
        heights = [float(b["accuracy"]) for b in bins]
        centers = [0.5 * (b["lower"] + b["upper"]) for b in bins]
        counts = [int(b["count"]) for b in bins]

        width = [u - l for l, u in zip(decimals[:-1], decimals[1:])]
        ax.bar(
            centers,
            heights,
            width=width,
            align="center",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="perfect")
        # Annotate the bin counts above every non-empty bar.
        top = max(max(heights), 1.0)
        for center, height, count in zip(centers, heights, counts):
            ax.annotate(
                f"n={count}",
                xy=(center, min(height + 0.03, 0.98)),
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="0.35",
            )
        ece = entry["payload"].get("overall", {}).get("ece", float("nan"))
        brier = entry["payload"].get("overall", {}).get("brier", float("nan"))
        ax.set_title(
            f"{entry['model_pretty']}\n{entry['split']} test — ECE={ece:.3f}, Brier={brier:.3f}",
            fontsize=9,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
    for index in range(n, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)

    # Shared axis labels.
    fig.supxlabel("mean predicted probability", fontsize=10)
    fig.supylabel("observed positive frequency", fontsize=10)
    fig.suptitle("Reliability diagrams (calibrated detector, test videos)", fontsize=11)
    fig.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot the reliability diagrams.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/figures/reliability.png")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    plot_reliability(args.results, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
