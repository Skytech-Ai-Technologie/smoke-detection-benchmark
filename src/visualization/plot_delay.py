"""Operating-point figure: detection delay per positive video vs false-alarm cost.

Three panels share one row per model (rows sorted by median delay):

* **(a)** time-to-detection CDF on a log axis: the empirical fraction of the
  pooled positive test videos detected *within* ``t`` seconds, per model.
  This recall-within-horizon view is exact under censoring (a video that
  ends without an alarm contributes to the denominator but never to the
  numerator). Vertical guides mark the protocol horizons (0.5 / 1 / 2 s).
* **(b)** the per-video delays behind (a): a box (IQR + median) over the
  detected delays, one jittered point per detected video coloured by split,
  and the missed videos as censored observations (x) drawn at that *video's
  duration* -- not at an arbitrary ceiling.
* **(c)** the false-alarm cost of the same operating point: mean false
  alarms per hour over the negative test videos (error bar = std over
  splits). Delay numbers alone are meaningless without this cost: a fast
  rule that alarms constantly would trivially "detect" everything.

Reading guide: a good model rises early in (a), sits left in (b) *and* is
cheap in (c); recall is the third coordinate and is annotated everywhere
(the proposed model in bold). A vector PDF copy is written next to the PNG.

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

# Okabe-Ito colour-blind-safe palette for the five test splits.
SPLIT_COLORS = {
    "split01": "#0072B2",
    "split02": "#E69F00",
    "split03": "#009E73",
    "split04": "#D55E00",
    "split05": "#CC79A7",
}

HORIZONS = (0.5, 1.0, 2.0)


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
    """Render the three-panel delay/recall/cost figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = collect_delays(results_dir)
    if not data:
        raise RuntimeError("no video-evaluation reports found")

    models = list(data)
    values = [np.asarray([v for v, _, _ in data[m]["obs"]], dtype=np.float64) for m in models]
    medians = [float(np.median(v)) if v.size else np.inf for v in values]
    order = sorted(range(len(models)), key=lambda i: medians[i])  # slot -> model index
    n_slots = len(order)

    # Per-model aggregates used by every panel.
    fa_mean, fa_std = {}, {}
    for m in models:
        fas = [fa for _, fa in data[m]["fa_per_split"]]
        fa_mean[m] = float(np.mean(fas)) if fas else float("nan")
        fa_std[m] = float(np.std(fas, ddof=1)) if len(fas) > 1 else 0.0
    max_fa_axis = max(fa_mean[m] + fa_std[m] for m in models if not np.isnan(fa_mean[m])) * 1.32

    fig = plt.figure(figsize=(11.8, 6.7))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 2.1], wspace=0.52,
        left=0.095, right=0.988, top=0.885, bottom=0.135,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.8, 0.66], wspace=0.09)
    ax_b = fig.add_subplot(inner[0, 0])
    ax_c = fig.add_subplot(inner[0, 1], sharey=ax_b)

    # Distinct muted colour per model so the CDF in (a) can be matched to the
    # row boxes in (b); the proposed TCN always in crimson.
    _CYCLE = ["#4C78A8", "#72B7B2", "#B279A2", "#FFBF79", "#54A24B", "#B79B5B", "#9D755D", "#8172B3"]
    model_line_color: dict[str, str] = {}
    for slot, idx in enumerate(order):
        model = models[idx]
        slot_col = _CYCLE[slot % len(_CYCLE)]
        model_line_color[model] = "#D62728" if model == "tcn" else slot_col

    # ------------------------------------------------------------------ (a)
    # Time-to-detection CDF: recall within t, exact under censoring.
    for slot, idx in enumerate(order):
        model = models[idx]
        obs = data[model]["obs"]
        n_total = len(obs)
        delays = np.sort(np.asarray([v for v, c, _ in obs if not c], dtype=np.float64))
        is_tcn = model == "tcn"
        color = model_line_color[model]
        lw = 2.6 if is_tcn else 1.4
        alpha = 1.0 if is_tcn else 0.75
        zorder = 6 if is_tcn else 3
        if delays.size == 0:
            continue
        x = np.clip(delays, 0.25, None)
        y = np.arange(1, delays.size + 1) / n_total
        y = np.concatenate([[0.0], y])
        x = np.concatenate([[x[0]], x])
        ax_a.step(x, y, where="post", color=color, linewidth=lw, alpha=alpha, zorder=zorder)
        # Endpoint marker: the final recall level of this model.
        ax_a.plot(
            [x[-1]], [y[-1]],
            marker="*" if is_tcn else "o",
            markersize=9 if is_tcn else 4.5,
            color=color, zorder=7, markeredgecolor="white", markeredgewidth=0.4,
        )
        r_at_2 = float((delays <= 2.0).sum()) / float(n_total)  # censored count in the denominator
        pretty = data[model]["model_pretty"]
        if model == "gru":
            pretty += "  (split01 only)"
        ax_a.plot([], [], color=color, linewidth=lw, alpha=alpha, solid_capstyle="butt",
                  label=f"{pretty} — {delays.size}/{n_total}, @2 s: {r_at_2:.0%}")
        # Tint the matching box in (b) with the same hue (drawn there).

    ax_a.set_xscale("log")
    ax_a.set_xlim(0.25, 150)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax_a.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax_a.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    for horizon in HORIZONS:
        ax_a.axvline(horizon, color="0.65", linestyle="--", linewidth=0.8, zorder=1)
    ax_a.set_xlabel("time since smoke onset,  t  (s, log scale)", fontsize=9)
    ax_a.set_ylabel("fraction of positive videos detected within t  (recall)", fontsize=9)
    ax_a.grid(alpha=0.22, which="major", linewidth=0.5)
    ax_a.legend(fontsize=6.8, loc="lower right", framealpha=0.95, handlelength=2.0)

    # ---------------------------------------------------------------- (b)+(c)
    # Rows share the axis in both panels; GRU's bar gets a dagger.
    censored_note_drawn = False
    for slot, idx in enumerate(order):
        model = models[idx]
        entry = data[model]
        obs = entry["obs"]
        y = slot
        is_tcn = model == "tcn"

        detected = np.asarray([v for v, c, _ in obs if not c], dtype=np.float64)

        # ---- (b) per-video delay distribution -----------------------------
        face = model_line_color[model]
        if detected.size >= 2:
            ax_b.boxplot(
                detected,
                vert=False,
                positions=[y],
                widths=0.46,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.3},
                boxprops={"facecolor": face, "alpha": 0.30 if is_tcn else 0.22},
                whiskerprops={"linewidth": 1.2 if is_tcn else 0.8,
                              "color": "#8C2D2D" if is_tcn else "0.42"},
                capprops={"linewidth": 1.2 if is_tcn else 0.8,
                          "color": "#8C2D2D" if is_tcn else "0.42"},
            )
        elif detected.size:
            ax_b.plot(detected, [y] * detected.size, "|", color="0.4", markersize=10, alpha=0.6)

        for i, split in enumerate(sorted(SPLIT_COLORS)):
            vals = [v for v, c, s in obs if not c and s == split]
            if not vals:
                continue
            jitter = (np.random.default_rng(idx * 10 + i).random(len(vals)) - 0.5) * 0.16
            ax_b.scatter(
                vals, y + jitter, s=24, alpha=0.85, edgecolors="none", zorder=4,
                color=SPLIT_COLORS[split],
                label=f"{split}" if slot == 0 else None,
            )
        censored = np.asarray([v for v, c, _ in obs if c], dtype=np.float64)
        if censored.size:
            jc = (np.random.default_rng(idx + 500).random(censored.size) - 0.5) * 0.16
            ax_b.scatter(
                censored, y + jc, marker="x", s=40, color="crimson",
                linewidths=1.4, alpha=0.75, zorder=3,
                label=None if censored_note_drawn else "missed — censored at video end",
            )
            censored_note_drawn = True

        # ---- (c) false-alarm cost -----------------------------------------
        if not np.isnan(fa_mean[model]):
            ax_c.barh(
                y, fa_mean[model], xerr=fa_std[model], height=0.55,
                color="#D62728" if is_tcn else "#EFA9A9",
                edgecolor="white", linewidth=0.4,
                error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": "0.3"},
            )
            ax_c.text(
                fa_mean[model] + fa_std[model] + 0.03 * max_fa_axis, y,
                f"{fa_mean[model]:.1f}" + ("  †" if model == "gru" else ""),
                va="center", ha="left", fontsize=7.5,
                color="0.1" if is_tcn else "0.3",
                fontweight="bold" if is_tcn else "normal",
            )
        else:
            ax_c.text(0.02 * max_fa_axis, y, "n/a", va="center", ha="left",
                      fontsize=7.5, color="0.4")

    for horizon in HORIZONS:
        ax_b.axvline(horizon, color="0.65", linestyle="--", linewidth=0.8, zorder=1)

    # Row labels under panel (b), sorted by median delay; the TCN in bold.
    labels: list[str] = []
    for i in order:
        model = models[i]
        obs = data[model]["obs"]
        n_det = sum(1 for _, c, _ in obs if not c)
        labels.append(
            f"{data[model]['model_pretty']}\n"
            f"median {medians[i]:.1f} s — {n_det}/{len(obs)}"
        )
    ax_b.set_yticks(range(n_slots))
    ax_b.set_yticklabels(labels, fontsize=7.5)
    for tick, i in zip(ax_b.get_yticklabels(), order):
        if models[i] == "tcn":
            tick.set_fontweight("bold")

    ax_b.set_xlabel("time to first alarm after smoke onset (s) — dashed guides: 0.5 / 1 / 2 s", fontsize=9)
    ax_c.set_xlabel("false alarms per hour\n(mean over splits)", fontsize=8)
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(-0.6, n_slots - 0.4)
    ax_c.set_xlim(left=0, right=max_fa_axis)
    ax_b.invert_yaxis()
    ax_b.grid(alpha=0.22, axis="x", linewidth=0.5)
    ax_c.grid(alpha=0.22, axis="x", linewidth=0.5)
    ax_c.tick_params(labelleft=False)

    # Panel letters and titles.
    ax_a.set_title("(a) recall within t — pooled positives", fontsize=9.5, loc="left")
    ax_b.set_title("(b) delay per video — detected (dots) vs censored (x)", fontsize=9.5, loc="left")
    ax_c.set_title("(c) false-alarm cost", fontsize=9.5, loc="left")

    handles_b, labels_b = ax_b.get_legend_handles_labels()
    if handles_b:
        fig.legend(handles_b, labels_b, fontsize=7, loc="lower center", ncol=6,
                   frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Detection delay at the operating point: speed, per-video distribution,"
        " and false-alarm cost   († GRU evaluated on split01 only)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    pdf_path = target.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s and %s", target, pdf_path)
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
