"""Operating-point figure: detection delay per positive video vs false-alarm cost.

Two panels share one row per model:

* **left** -- time to first alarm of the positive videos, pooled over the
  test splits (box over detected delays, one jittered point per video
  coloured by split); missed videos are drawn as censored observations (✗)
  at that *video's duration* -- not at an arbitrary ceiling. Vertical guides
  mark the detection-delay horizons of the protocol (0.5 / 1 / 2 s).
* **right** -- the false-alarm cost of the same operating point: mean false
  alarms per hour over the negative test videos (error bar = std over
  splits). The delay numbers alone are meaningless without this cost: a
  fast rule that alarms constantly would trivially "detect" everything.

Reading guide, therefore: a good model is *fast* in the left panel *and*
*short* (cheap) in the right panel. The number of detected positives is
annotated per row, because recall is the third coordinate of the trade-off.

Example
-------
    python -m src.visualization.plot_delay --results results/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger, setup_logging
from .labels import parse_eval_artifact

__all__ = ["collect_delays", "plot_delay", "main"]

LOGGER = get_logger("visualization.plot_delay")

TARGET_LABEL = "TCN (proposed)"


def collect_delays(results_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Collect per-video alarm times and per-split alarm cost, per model.

    Returns ``{model_tag: {"obs": [(value, censored, split), ...],
    "fa_per_split": [(split, fa_per_hour), ...], "model_pretty": str}}``.
    """
    by_model: dict[str, dict[str, Any]] = {}
    results_dir = Path(results_dir)
    for path in sorted(results_dir.glob("video_eval_*.json")):
        identity = parse_eval_artifact(path.stem)
        if identity is None:
            LOGGER.warning("skipping %s (unparsable identity)", path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "video_metrics" not in payload or "per_video" not in payload:
            continue
        entry = by_model.setdefault(
            identity["model"],
            {"obs": [], "fa_per_split": [], "model_pretty": identity["model_pretty"]},
        )
        for row in payload["per_video"]:
            if row.get("label") != "smoke":
                continue
            duration = float(row.get("duration_sec") or 0.0)
            if row.get("detected") and row.get("detection_delay") is not None:
                entry["obs"].append((float(row["detection_delay"]), False, identity["split"]))
            else:
                # Censored: the video ended without an alarm.
                entry["obs"].append((duration, True, identity["split"]))
        fa = payload["video_metrics"].get("false_alarms_per_hour")
        if fa is not None:
            entry["fa_per_split"].append((identity["split"], float(fa)))
    return dict(sorted(by_model.items()))


def plot_delay(results_dir: str | Path, out_path: str | Path) -> Path:
    """Render the delay distribution and false-alarm cost per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = collect_delays(results_dir)
    if not data:
        raise RuntimeError("no video-evaluation reports found")

    models = list(data)
    values = [np.asarray([v for v, _, _ in data[m]["obs"]], dtype=np.float64) for m in models]
    medians = [float(np.median(v)) if v.size else np.inf for v in values]
    order = sorted(range(len(models)), key=lambda i: medians[i])
    max_upper = max(
        float(np.mean([fa for _, fa in data[m]["fa_per_split"]]))
        + (float(np.std([fa for _, fa in data[m]["fa_per_split"]], ddof=1))
           if len(data[m]["fa_per_split"]) > 1 else 0.0)
        for m in models if data[m]["fa_per_split"]
    )
    max_fa = max(
        float(np.mean([fa for _, fa in data[m]["fa_per_split"]]))
        for m in models if data[m]["fa_per_split"]
    )

    fig = plt.figure(figsize=(9.6, 0.62 * len(models) + 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.05], wspace=0.06)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1], sharey=ax_d)

    split_colors = {f"split0{i + 1}": f"C{i}" for i in range(5)}
    delay_handles = []
    fa_labels_drawn = False
    for slot, idx in enumerate(order):
        model = models[idx]
        entry = data[model]
        obs = entry["obs"]
        y = slot
        delays = np.asarray([v for v, c, _ in obs if not c], dtype=np.float64)
        n_total = len(obs)
        n_det = int(delays.size)

        # ---- left panel: delay distribution --------------------------------
        if delays.size >= 2:
            ax_d.boxplot(
                delays,
                vert=False,
                positions=[y],
                widths=0.5,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"facecolor": "0.85", "alpha": 0.55},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )
        elif delays.size:
            ax_d.plot(delays, [y] * delays.size, "|", color="0.4", markersize=10, alpha=0.6)
        for i, split in enumerate(sorted(split_colors)):
            vals = [v for v, c, s in obs if not c and s == split]
            if not vals:
                continue
            jitter = (np.random.default_rng(idx * 10 + i).random(len(vals)) - 0.5) * 0.17
            ax_d.scatter(
                vals, y + jitter, s=22, alpha=0.8, edgecolors="none", zorder=3,
                color=split_colors[split],
                label=f"{split}" if model == models[order[0]] else None,
            )
        censored = np.asarray([v for v, c, _ in obs if c], dtype=np.float64)
        if censored.size:
            jc = (np.random.default_rng(idx + 500).random(censored.size) - 0.5) * 0.17
            ax_d.scatter(
                censored, y + jc, marker="x", s=46, color="crimson", linewidths=1.6,
                zorder=3, label=None if fa_labels_drawn else "missed — censored at video end",
            )
            fa_labels_drawn = True

        # ---- right panel: false-alarm cost ---------------------------------
        fas = [fa for _, fa in entry["fa_per_split"]]
        if fas:
            mean_fa = float(np.mean(fas))
            std_fa = float(np.std(fas, ddof=1)) if len(fas) > 1 else 0.0
            bar = ax_f.barh(
                y, mean_fa, xerr=std_fa, height=0.55,
                color="#f4a6a6" if model != "tcn" else "#d62728",
                edgecolor="white", linewidth=0.4,
                error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": "0.3"},
            )
            ax_f.text(
                mean_fa + std_fa + 0.03 * max_upper, y,
                f"{mean_fa:.1f}" + ("" if model != "gru" else " †"),
                va="center", ha="left", fontsize=7.5, color="0.25",
            )
        else:
            ax_f.text(0.02 * max_upper, y, "n/a", va="center", ha="left",
                      fontsize=7.5, color="0.4")

    # Detection-delay horizons of the protocol (guides; explained in the axis label).
    for horizon in (0.5, 1.0, 2.0):
        ax_d.axvline(horizon, color="0.6", linestyle="--", linewidth=0.8, zorder=1)

    # Shared y labels, sorted by median delay; the proposed model in bold.
    labels: list[str] = []
    for i in order:
        model = models[i]
        obs = data[model]["obs"]
        n_det = sum(1 for _, c, _ in obs if not c)
        labels.append(
            f"{data[model]['model_pretty']}\n"
            f"median {medians[i]:.1f} s — recall {n_det}/{len(obs)}"
        )
    ax_d.set_yticks(range(len(order)))
    ax_d.set_yticklabels(labels, fontsize=7.5)
    for tick, i in zip(ax_d.get_yticklabels(), order):
        if models[i] == "tcn":
            tick.set_fontweight("bold")

    ax_d.set_xlabel(
        "time to first alarm after smoke onset (s) — dashed guides: 0.5 / 1 / 2 s horizons"
    )
    ax_f.set_xlabel("false alarms per hour\n(mean over test splits)", fontsize=8)
    ax_f.set_title("cost", fontsize=9)
    ax_d.set_title("speed — pooled over the test splits", fontsize=9, loc="left")
    ax_f.tick_params(labelleft=False)
    ax_d.invert_yaxis()
    ax_d.set_xlim(left=0)
    ax_f.set_xlim(left=0, right=max_upper * 1.35)
    ax_d.grid(alpha=0.25, axis="x", linewidth=0.5)
    ax_f.grid(alpha=0.25, axis="x", linewidth=0.5)
    fig.suptitle(
        "Operating point per model: how fast is the alarm, and what does it cost?"
        "  († evaluated on one split only)",
        fontsize=10,
    )
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    if handles_d:
        fig.legend(handles_d, labels_d, fontsize=7, loc="lower center", ncol=6,
                   frameon=False, bbox_to_anchor=(0.42, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot the detection-delay distributions.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/figures/delay.png")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    plot_delay(args.results, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
