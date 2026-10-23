"""ROC curves for every evaluated model.

Two figures are produced (``out`` and ``out`` with ``_per_split`` inserted
before the extension):

* **pooled** -- one ROC per *model*, pooling the frame scores of every test
  split in which the model was evaluated. This is the headline figure: it
  answers "how well does each model discriminate frames?" and the curves are
  directly comparable because they are each built on the same pooled videos.
* **per split** -- small multiples, one panel per split, each showing the
  models evaluated on that split. Test sets differ between panels, so the
  panels may not be ranked against each other.

Frame labels: a frame is positive when the video's smoke onset has been
reached (see ``src.data.extract_frames``). The frame level is deliberately
the *harder* level; the video level is covered by the sweep figures.

Example
-------
    python -m src.visualization.plot_roc --results results/raw --out results/figures/roc.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger, setup_logging
from .labels import parse_eval_artifact

__all__ = ["collect_curves", "plot_roc", "main"]

LOGGER = get_logger("visualization.plot_roc")


def collect_curves(results_dir: str | Path) -> dict[str, dict]:
    """Load every ``*_frames.npz`` archive with its parsed identity.

    Legacy artifacts (``frame_eval_splitNN_...``, produced before the
    ``*_tcn_*`` renaming) are dropped when a tagged artifact of the same
    model and split exists, so the same curve never appears twice.
    """
    entries: list[dict] = []
    results_dir = Path(results_dir)
    for path in sorted(results_dir.glob("*_frames.npz")):
        identity = parse_eval_artifact(path.stem.replace("_frames", ""))
        if identity is None:
            LOGGER.warning("skipping %s (not a per-split frame evaluation)", path.name)
            continue
        with np.load(path) as data:
            scores = data["scores"]
            labels = data["labels"]
        if labels.size == 0 or len(np.unique(labels)) < 2:
            LOGGER.warning("skipping %s (single class)", path.name)
            continue
        identity["scores"] = np.asarray(scores, dtype=np.float64)
        identity["labels"] = np.asarray(labels, dtype=np.int8)
        entries.append(identity)

    # Drop legacy duplicates of the same (model, split).
    by_key: dict[tuple[str, str, str], dict] = {}
    for entry in entries:
        key = (entry["kind"], entry["model"], entry["split"])
        if key in by_key and not entry["legacy"]:
            by_key[key] = entry  # tagged artifact wins over the legacy one
        else:
            by_key.setdefault(key, entry)
    return {
        f"{entry['model']}@{entry['split']}": entry for key, entry in sorted(by_key.items())
    }


def _style(model_tag: str) -> dict:
    palette = {
        "tcn": ("C0", 2.2),
        "gru": ("C9", 1.8),
        "spatial": ("C1", 1.8),
        "evidence": ("C4", 1.8),
    }
    color, width = palette.get(model_tag, (None, 1.5))
    return {"color": color, "linewidth": width}


def plot_pooled(
    entries: dict[str, dict],
    out_path: Path,
) -> Path:
    """One ROC per model, pooling every split in which the model was evaluated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

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
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = float(np.trapezoid(tpr, fpr))
        style = _style(model_tag)
        ax.plot(
            fpr,
            tpr,
            label=f"{pretty[model_tag]} (AUC={auroc:.3f}, n={labels.size})",
            **style,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Frame-level ROC — pooled over the test splits")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def plot_per_split(
    entries: dict[str, dict],
    out_path: Path,
) -> Path:
    """Small multiples: one panel per split, one curve per evaluated model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    splits = sorted({entry["split"] for entry in entries.values()})
    n = len(splits)
    n_cols = min(3, max(1, n))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows), squeeze=False)

    # One shared legend across the panels (models share colors everywhere).
    seen_models: list[str] = []
    for index, split in enumerate(splits):
        ax = axes[index // n_cols][index % n_cols]
        panel_entries = sorted(
            (entry for entry in entries.values() if entry["split"] == split),
            key=lambda e: (e["model"],),
        )
        lines: list[tuple[str, object]] = []
        for entry in panel_entries:
            fpr, tpr, _ = roc_curve(entry["labels"], entry["scores"])
            auroc = float(np.trapezoid(tpr, fpr))
            style = _style(entry["model"])
            line = ax.plot(fpr, tpr, **style)[0]
            lines.append((f"{entry['model_pretty']} {auroc:.2f}", line))
            if entry["model"] not in seen_models:
                seen_models.append(entry["model"])
        # Inside-panel annotation of the AUC per model, stacked top-left.
        for slot, (label_text, _line) in enumerate(lines):
            ax.annotate(
                label_text,
                xy=(0.02, 0.97 - 0.09 * slot),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=6.5,
                color=_line.get_color(),
            )
        ax.set_title(f"{split} (test)", fontsize=9)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
    # Hide unused panels.
    for index in range(n, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)
    fig.suptitle("Frame-level ROC per split (test sets differ between panels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def plot_roc(results_dir: str | Path, out_path: str | Path) -> Path:
    """Render the pooled ROC figure and the per-split companion."""
    entries = collect_curves(results_dir)
    if not entries:
        raise RuntimeError("no valid frame-evaluation archives found")
    target = Path(out_path)
    plot_pooled(entries, target)
    stem, suffix = target.stem, target.suffix
    parent = target.parent
    plot_per_split(entries, parent / f"{stem}_per_split{suffix}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot the ROC curves.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/figures/roc.png")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    plot_roc(args.results, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
