"""Precision-recall curves for every evaluated model.

Two figures (``out`` and ``out`` with ``_per_split`` before the extension):

* **pooled** -- one PR curve per *model*, pooling the frame scores of every
  test split in which the model was evaluated (same pooling as
  ``plot_roc``); curves are directly comparable.
* **per split** -- small multiples, one panel per split. Test sets differ
  between panels, so panels may not be ranked against each other.

Frame labels: a frame is positive when the video's smoke onset has been
reached. The average precision (AP) is shown in the legend / annotation.

Example
-------
    python -m src.visualization.plot_pr --results results/raw --out results/figures/pr.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger, setup_logging
from .labels import parse_eval_artifact

__all__ = ["collect_curves", "plot_pr", "main"]

LOGGER = get_logger("visualization.plot_pr")


def collect_curves(results_dir: str | Path) -> dict[str, dict]:
    """Load every ``*_frames.npz`` archive with its parsed identity.

    Legacy artifacts are dropped when a tagged artifact of the same
    model and split exists (see :mod:`plot_roc`).
    """
    entries: list[dict] = []
    for path in sorted(Path(results_dir).glob("*_frames.npz")):
        identity = parse_eval_artifact(path.stem.replace("_frames", ""))
        if identity is None:
            LOGGER.warning("skipping %s (not a per-split frame evaluation)", path.name)
            continue
        with np.load(path) as data:
            scores = data["scores"]
            labels = data["labels"]
        if labels.size == 0 or len(np.unique(labels)) < 2 or labels.sum() == 0:
            LOGGER.warning("skipping %s (degenerate labels)", path.name)
            continue
        identity["scores"] = np.asarray(scores, dtype=np.float64)
        identity["labels"] = np.asarray(labels, dtype=np.int8)
        entries.append(identity)

    by_key: dict[tuple[str, str, str], dict] = {}
    for entry in entries:
        key = (entry["kind"], entry["model"], entry["split"])
        if key in by_key and not entry["legacy"]:
            by_key[key] = entry
        else:
            by_key.setdefault(key, entry)
    return {f"{entry['model']}@{entry['split']}": entry for _, entry in sorted(by_key.items())}


def _style(model_tag: str) -> dict:
    palette = {
        "tcn": ("C0", 2.2),
        "gru": ("C9", 1.8),
        "spatial": ("C1", 1.8),
        "evidence": ("C4", 1.8),
    }
    color, width = palette.get(model_tag, (None, 1.5))
    return {"color": color, "linewidth": width}


def plot_pooled(entries: dict[str, dict], out_path: Path) -> Path:
    """One PR curve per model, pooling every split in which the model ran."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    pretty: dict[str, str] = {}
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    for entry in entries.values():
        pretty.setdefault(entry["model"], entry["model_pretty"])
        slot = pooled.setdefault(entry["model"], {"scores": [], "labels": []})
        slot["scores"].append(entry["scores"])
        slot["labels"].append(entry["labels"])

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for model_tag, data in sorted(pooled.items()):
        scores = np.concatenate(data["scores"])
        labels = np.concatenate(data["labels"])
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = float(average_precision_score(labels, scores))
        style = _style(model_tag)
        ax.plot(
            recall,
            precision,
            label=f"{pretty[model_tag]} (AP={ap:.3f}, n={labels.size})",
            **style,
        )
    all_labels: list[np.ndarray] = []
    for data in pooled.values():
        all_labels.extend(data["labels"])
    baseline = float(np.concatenate(all_labels).mean()) if all_labels else 0.0
    ax.axhline(baseline, color="k", linestyle="--", linewidth=0.8,
               label=f"chance ({baseline:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title("Frame-level precision-recall — pooled over the test splits")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def plot_per_split(entries: dict[str, dict], out_path: Path) -> Path:
    """Small multiples: one panel per split, one curve per evaluated model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    splits = sorted({entry["split"] for entry in entries.values()})
    n = len(splits)
    n_cols = min(3, max(1, n))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows), squeeze=False)

    for index, split in enumerate(splits):
        ax = axes[index // n_cols][index % n_cols]
        panel_entries = sorted(
            (entry for entry in entries.values() if entry["split"] == split),
            key=lambda e: e["model"],
        )
        annotations: list[tuple[str, object]] = []
        for entry in panel_entries:
            precision, recall, _ = precision_recall_curve(entry["labels"], entry["scores"])
            ap = float(average_precision_score(entry["labels"], entry["scores"]))
            style = _style(entry["model"])
            line = ax.plot(recall, precision, **style)[0]
            annotations.append((f"{entry['model_pretty']} {ap:.2f}", line))
        for slot, (label_text, line) in enumerate(annotations):
            ax.annotate(
                label_text,
                xy=(0.98, 0.97 - 0.09 * slot),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=6.5,
                color=line.get_color(),
            )
        ax.set_title(f"{split} (test)", fontsize=9)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
    for index in range(n, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)
    fig.suptitle("Frame-level precision-recall per split (test sets differ between panels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def plot_pr(results_dir: str | Path, out_path: str | Path) -> Path:
    """Render the pooled PR figure and the per-split companion."""
    entries = collect_curves(results_dir)
    if not entries:
        raise RuntimeError("no valid frame-evaluation archives found")
    target = Path(out_path)
    plot_pooled(entries, target)
    stem, suffix = target.stem, target.suffix
    plot_per_split(entries, target.parent / f"{stem}_per_split{suffix}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot the PR curves.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/figures/pr.png")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    plot_pr(args.results, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
