"""Generate the paper's architecture diagrams (inference pipeline + benchmark flow).

Outputs PNGs into paper_package/architecture. Pure matplotlib drawing, no
external diagram tool required, so the figures stay regenerable.

    python tools/make_architecture.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "paper_package" / "architecture"


def _box(ax, xy, w, h, text, *, fc="#eef3fb", ec="#2b5ea7", fs=8, bold=False):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.1,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", linespacing=1.35,
    )
    return (x + w, y + h / 2)  # right anchor


def _arrow(ax, start, end, *, style="-|>", color="0.25", ls="-", lw=1.2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, color=color, linestyle=ls,
            linewidth=lw, mutation_scale=12, shrinkA=1, shrinkB=1,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def inference_pipeline() -> Path:
    """Classical front end + calibrated causal TCN + uncertain-aware policy."""
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    YA = 5.4   # proposed row A (front end + CNN + calibration)
    YB = 3.5   # proposed row B (TCN → policy → alarm)
    YC = 1.6   # baseline row

    ax.text(0.2, YA + 1.25, "Proposed detector — calibrated causal TCN over candidate crops",
            fontsize=11, fontweight="bold")

    # ---- row A: shared front end → CNN → calibration ------------------------
    _box(ax, (0.3, YA), 1.2, 1.0, "video frame\n(FIRESENSE)", fc="#f6f6f6", ec="0.4")
    _box(ax, (1.9, YA), 1.8, 1.0, "Kalman background\n(constant velocity)", fc="#fdf3e3", ec="#b8761e")
    _box(ax, (4.1, YA), 1.8, 1.0, "HSV colour filter\n+ area / aspect rules", fc="#fdf3e3", ec="#b8761e")
    _box(ax, (6.3, YA), 1.8, 1.0, "candidate crops\n≤ 8 per frame,\n50×50 px", fc="#fdf3e3", ec="#b8761e")
    cnn = _box(ax, (8.5, YA), 2.1, 1.0, "spatial CNN\n(MIL-trained,\nmax over candidates)", fc="#e8f4ec", ec="#2e7d43")
    _box(ax, (11.0, YA), 1.7, 1.0, "isotonic calibration\n(fit on val)", fc="#e8eaf7", ec="#4a4a8f")
    _arrow(ax, (1.5, YA + 0.5), (1.9, YA + 0.5))
    _arrow(ax, (3.7, YA + 0.5), (4.1, YA + 0.5))
    _arrow(ax, (5.9, YA + 0.5), (6.3, YA + 0.5))
    _arrow(ax, (8.1, YA + 0.5), (8.5, YA + 0.5))
    _arrow(ax, (10.6, YA + 0.5), (11.0, YA + 0.5))

    # ---- row B: TCN → (reliability) → policy → alarm ------------------------
    _box(ax, (10.35, YB), 2.35, 1.0, "causal TCN\n3 s window,\nstreamable", fc="#e8f4ec", ec="#2e7d43", bold=True)
    _box(ax, (6.9, YB), 2.8, 1.0, "uncertainty-aware policy\nthreshold + persistence\n+ hysteresis", fc="#f3e6ee", ec="#96427a")
    _box(ax, (4.5, 4.45), 1.9, 0.85, "reliability model\n(6 features)", fc="#f3e6ee", ec="#96427a", fs=7.5)
    _box(ax, (2.0, YB), 2.0, 1.0, "ALARM\n(+ delay budget)", fc="#dfe7f5", ec="#2b5ea7", bold=True)
    _arrow(ax, (11.85, YA), (11.5, YB + 1.0))                                # calibration → TCN
    _arrow(ax, (10.35, YB + 0.5), (9.7, YB + 0.5))                           # TCN → policy
    _arrow(ax, (9.3, YA), (5.9, 5.32), ls="--", color="0.5", lw=1.0)         # CNN → reliability
    _arrow(ax, (5.45, 4.45), (7.2, 4.5))                                     # reliability → policy
    _arrow(ax, (6.9, YB + 0.5), (2.0 + 2.0, YB + 0.5))                       # policy → alarm

    # ---- row C: classical baseline -----------------------------------------
    ax.text(0.2, YC + 1.45, "Classical baseline — frame-only CNN with a fixed threshold",
            fontsize=11, fontweight="bold", color="0.45")
    _box(ax, (4.5, YC), 2.0, 1.0, "same frame\n(no temporal p)", fc="#f2f2f2", ec="0.6")
    _box(ax, (7.3, YC), 2.4, 1.0, "spatial CNN on\n1 candidate only", fc="#f2f2f2", ec="0.6")
    _box(ax, (10.5, YC), 2.1, 1.0, "fixed threshold\np ≥ 0.70 (1 frame)", fc="#f2f2f2", ec="0.6")
    _box(ax, (12.85, YC), 1.05, 1.0, "ALARM", fc="#e0e0e0", ec="0.6")
    _arrow(ax, (4.5 + 2.0, YC + 0.5), (7.3, YC + 0.5))
    _arrow(ax, (7.3 + 2.4, YC + 0.5), (10.5, YC + 0.5))
    _arrow(ax, (10.5 + 2.1, YC + 0.5), (12.85, YC + 0.5))

    # note: shared front end
    ax.text(
        0.3, 0.85,
        "Controlled comparison: every model shares the same Kalman/HSV front end; only the\n"
        "temporal-decision stage (rows A→B) is swapped in for the proposed detector.",
        fontsize=8, color="0.3", va="top", linespacing=1.6,
    )
    target = OUT_DIR / "architecture_inference_pipeline.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def benchmark_flow() -> Path:
    """Data → splits → train/val/test → evaluation modes → statistics."""
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    row_y = 2.55       # bottom of the main row
    row_h = 1.25       # height of the main row

    x1 = 0.2
    _box(ax, (x1, row_y), 1.9, row_h, "FIRESENSE\n22 videos\n(9 smoke · 13 negative)\nsession-level ids", fc="#f6f6f6", ec="0.4", fs=7.5)
    x2 = 2.55
    s = _box(ax, (x2, row_y), 2.25, row_h, "5 predefined\nvideo-level splits\nno frame/session leakage\ntest manifests locked", fc="#eef3fb", ec="#2b5ea7", bold=True, fs=7.5)
    _arrow(ax, (x1 + 1.9, row_y + row_h / 2), (x2, row_y + row_h / 2))

    x3 = 5.25
    # one wide box holding the three roles
    ax.add_patch(
        FancyBboxPatch(
            (x3, row_y), 4.45, row_h,
            boxstyle="round,pad=0.06,rounding_size=0.06",
            facecolor="#fbfbfb", edgecolor="0.45", linewidth=1.1,
        )
    )
    ax.text(x3 + 2.22, row_y + row_h + 0.10, "each split is used exactly once, in this order",
            ha="center", fontsize=7.5, color="0.35")
    w_inner = 1.28
    _box(ax, (x3 + 0.10, row_y + 0.18), w_inner, row_h - 0.36, "train\nfit CNN, TCN,\nstandardiser", fc="#e8f4ec", ec="#2e7d43", fs=6.5)
    _box(ax, (x3 + 1.52, row_y + 0.18), w_inner, row_h - 0.36, "val\nisotonic calibration,\nfa-budget thresholds", fc="#e8eaf7", ec="#4a4a8f", fs=6.5)
    _box(ax, (x3 + 2.94, row_y + 0.18), w_inner, row_h - 0.36, "test\nread-only,\ntouched once", fc="#fdeaea", ec="#b3272d", fs=6.5, bold=True)
    _arrow(ax, (x2 + 2.25, row_y + row_h / 2), (x3, row_y + row_h / 2))
    _arrow(ax, (x3 + 0.10 + w_inner / 2, row_y + row_h / 2 - 0.35), (x3 + 1.52 + w_inner / 2, row_y + row_h / 2 - 0.35))
    _arrow(ax, (x3 + 1.52 + w_inner / 2, row_y + row_h / 2 - 0.35), (x3 + 2.94 + w_inner / 2, row_y + row_h / 2 - 0.35))

    x4 = 9.95
    m = _box(ax, (x4, row_y), 1.85, row_h, "4 evaluation modes\nframe · video · streaming\nedge (pre-decoded)", fc="#fdf3e3", ec="#b8761e", fs=7)
    _arrow(ax, (x3 + 4.45, row_y + row_h / 2), (x4, row_y + row_h / 2))

    x5 = 12.15
    st = _box(ax, (x5, row_y), 0.95, row_h, "stats\nbootstrap\nunit=video", fc="#e8f4ec", ec="#2e7d43", fs=6.5)
    _arrow(ax, (x4 + 1.85, row_y + row_h / 2), (x5, row_y + row_h / 2))

    ax.text(
        0.2, 0.95,
        "Operating point is frozen on validation (isotonic map + alarm thresholds under a ≤ 1 false alarm/h budget);\n"
        "the test split is evaluated a single time per model.\n\n"
        "Outputs → paper: Table 1 (video accuracy) · Table 2 (edge latency/FPS/energy) · Table 3 (FP16/INT8 quantisation)\n"
        "· Table 4 (cross-dataset — empty, documented limitation) · Table 5 (ablations) · summary over splits (95 % CIs)\n"
        "· ROC / PR / reliability / delay / qualitative figures.",
        fontsize=8, color="0.25", linespacing=1.6, va="top",
    )
    target = OUT_DIR / "architecture_benchmark_flow.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = inference_pipeline()
    p2 = benchmark_flow()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
