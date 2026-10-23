"""Cross-dataset frame-level transfer of the spatial CNN to DFire (still images).

DFire contains standalone images with YOLO-format smoke/fire labels, so the
*video*-level claims of the benchmark cannot be evaluated on it (no streams,
no onsets, no FA/h). What can be evaluated is the **spatial component**:

* the split01 spatial CNN (never retrained, never re-calibrated on DFire —
  test-only protocol) scores 50×50 tiles of every image;
* a MIL-style max over tiles gives one probability per image;
* metrics: AUROC / AP for ``smoke`` vs ``none`` and, separately, for
  ``smoke`` vs the *fire distractor* group (flame images without smoke).

The temporal stage, delay, false-alarms-per-hour and edge claims stay
FIRESENSE-only by construction.

    python tools/eval_dfire_generalization.py \
        --dfire-root "D:\FORFIRE CODES\forfire1\wildfire-fusion\data\dfire" \
        --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["main"]


def _parse_label(label_path: Path) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    """YOLO label file -> (class ids, boxes). An empty file means background."""
    classes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        classes.append(c)
        boxes.append(tuple(float(v) for v in parts[1:5]))  # type: ignore[misc]
    return classes, boxes


def _collect_images(dfire_root: Path, max_per_group: int, seed: int) -> list[dict]:
    """Sample the three evaluation groups with a fixed seed."""
    images_dir = dfire_root / "images" / "test"
    labels_dir = dfire_root / "labels" / "test"
    groups: dict[str, list[dict]] = {"smoke": [], "fire_only": [], "none": []}
    for image_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        label_path = labels_dir / (image_path.stem + ".txt")
        classes, boxes = _parse_label(label_path) if label_path.exists() else ([], [])
        has_smoke = 0 in classes
        has_fire = 1 in classes
        if has_smoke:
            group = "smoke"
        elif has_fire:
            group = "fire_only"
        else:
            group = "none"
        groups[group].append(
            {
                "path": str(image_path),
                "name": image_path.name,
                "classes": sorted(set(classes)),
                "boxes": boxes,
                "group": group,
            }
        )
    rng = np.random.default_rng(seed)
    sampled: list[dict] = []
    for group in ("smoke", "none", "fire_only"):
        pool = groups[group]
        take = min(len(pool), max_per_group)
        idx = list(rng.choice(len(pool), size=take, replace=False))
        sampled.extend(pool[i] for i in idx)
        print(f"  {group}: {len(pool)} available -> {take} sampled", file=sys.stderr)
    return sampled


def _image_tiles(image: np.ndarray, tile: int, stride: int, max_tiles: int, rng) -> np.ndarray:
    """K×K sliding-window tiles of an image (no motion front end needed)."""
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        tiles = np.pad(image, ((0, tile - h), (0, tile - w), (0, 0)))
        return tiles[None]
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    y_grid = np.minimum(np.asarray(ys), h - tile)
    x_grid = np.minimum(np.asarray(xs), w - tile)
    tiles = [
        image[int(y): int(y) + tile, int(x): int(x) + tile]
        for y in y_grid
        for x in x_grid
    ]
    arr = np.asarray(tiles)
    if arr.shape[0] > max_tiles:
        pick = rng.choice(arr.shape[0], size=max_tiles, replace=False)
        arr = arr[np.sort(pick)]
    return arr


def main(argv: list[str] | None = None) -> int:
    """Run the DFire frame transfer evaluation, write reports and figure."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dfire-root",
        default=r"D:\FORFIRE CODES\forfire1\wildfire-fusion\data\dfire",
    )
    parser.add_argument("--spatial-checkpoint", default="results/raw/spatial_split01_seed42/model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-group", type=int, default=1500)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-tiles-per-image", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    import cv2
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    from src.models.builder import build_spatial_cnn
    from src.models.spatial_cnn import preprocess_crops
    from src.utils.logging import get_logger, setup_logging

    LOGGER = get_logger("tools.eval_dfire_generalization")
    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    ckpt = torch.load(args.spatial_checkpoint, map_location="cpu", weights_only=False)
    spatial_model = build_spatial_cnn(ckpt["meta"].get("model_config", {}))
    spatial_model.load_state_dict(ckpt["model"])
    input_size = tuple(int(v) for v in ckpt["meta"].get("model_config", {}).get("input_size", [50, 50]))
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    spatial_model.to(device).eval()
    LOGGER.info("spatial CNN -> device=%s, input_size=%s", device, input_size)

    dfire_root = Path(args.dfire_root)
    LOGGER.info("DFire root: %s", dfire_root)
    samples = _collect_images(dfire_root, args.max_images_per_group, args.seed)
    LOGGER.info("evaluating %d images", len(samples))

    rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    batch_tiles: list[np.ndarray] = []
    for i, sample in enumerate(samples):
        image = cv2.imread(sample["path"])
        if image is None:
            LOGGER.warning("unreadable image skipped: %s", sample["path"])
            continue
        tiles = _image_tiles(image, input_size[0], args.stride, args.max_tiles_per_image, rng)
        probs: list[float] = []
        for start in range(0, tiles.shape[0], args.batch_size):
            batch = preprocess_crops(tiles[start: start + args.batch_size], input_size)
            batch = batch.to(device)
            with torch.no_grad():
                p = spatial_model.predict_proba(batch)
            probs.extend(torch.sigmoid(p).float().cpu().tolist())
        rows.append(
            {
                "name": sample["name"],
                "path": sample["path"],
                "group": sample["group"],
                "label": 1 if sample["group"] == "smoke" else 0,
                "max_prob": float(max(probs)),
                "mean_prob": float(np.mean(probs)),
                "p90_prob": float(np.percentile(probs, 90)),
                "n_tiles": len(probs),
                "n_gt_objects": len(sample["classes"]),
            }
        )
        if (i + 1) % 250 == 0:
            LOGGER.info("scored %d / %d", i + 1, len(samples))

    smoke_mask = np.asarray([r["group"] == "smoke" for r in rows])
    none_mask = np.asarray([r["group"] == "none" for r in rows])
    fire_mask = np.asarray([r["group"] == "fire_only" for r in rows])

    def _auc(neg_mask: np.ndarray, key: str) -> dict:
        scores_key = np.asarray([r[key] for r in rows], dtype=np.float64)
        labels = np.concatenate([np.ones(int(smoke_mask.sum())), np.zeros(int(neg_mask.sum()))])
        sc = np.concatenate([scores_key[smoke_mask], scores_key[neg_mask]])
        return {
            "auroc": float(roc_auc_score(labels, sc)),
            "ap": float(average_precision_score(labels, sc)),
            "n_pos": int(smoke_mask.sum()),
            "n_neg": int(neg_mask.sum()),
        }

    def _group_distrib(key: str, group_mask: np.ndarray) -> dict:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)[group_mask]
        if values.size == 0:
            return {}
        return {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "frac_ge_0_5": float((values >= 0.5).mean()),
        }

    # Two tile aggregations, because max-over-400-tiles saturates by design.
    aggregations = {}
    for key, note in (
        ("max_prob", "MIL max over tiles (saturating by design)"),
        ("mean_prob", "mean over tiles of the whole image"),
    ):
        aggregations[key] = {
            "aggregation": note,
            "auroc_smoke_vs_none": _auc(none_mask, key),
            "auroc_smoke_vs_fire_only": _auc(fire_mask, key) if fire_mask.any() else None,
            "score_distribution": {
                "smoke": _group_distrib(key, smoke_mask),
                "none": _group_distrib(key, none_mask),
                "fire_only": _group_distrib(key, fire_mask),
            },
        }

    metrics = {
        "protocol": "50x50 tiles (stride 25, <=400/image); no retrain, no recalibration; DFire = test-only",
        "checkpoint": args.spatial_checkpoint,
        "dfire_root": str(dfire_root),
        "seed": args.seed,
        "n_images": len(rows),
        "aggregations": aggregations,
    }

    out_dir = ROOT / "results" / "raw" / "dfire_frame_transfer_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "path", "group", "label", "max_prob", "mean_prob",
                        "p90_prob", "n_tiles", "n_gt_objects"],
        )
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote %s and per_image.csv", out_dir / "metrics.json")

    # --- figure: transfer ROC ------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), squeeze=False)
    for col, key in enumerate(("max_prob", "mean_prob")):
        ax = axes[0][col]
        for neg_mask, color, comp_name in (
            (none_mask, "C0", "smoke vs none"),
            (fire_mask, "C1", "smoke vs fire distractors"),
        ):
            if not neg_mask.any():
                continue
            labels = np.concatenate(
                [np.ones(int(smoke_mask.sum())), np.zeros(int(neg_mask.sum()))]
            )
            scores_key = np.asarray([r[key] for r in rows], dtype=np.float64)
            sc = np.concatenate([scores_key[smoke_mask], scores_key[neg_mask]])
            fpr, tpr, _ = roc_curve(labels, sc)
            auroc = float(roc_auc_score(labels, sc))
            ax.plot(
                fpr, tpr, color=color, linewidth=2.0,
                label=f"{comp_name} — AUC {auroc:.3f}\n({int(smoke_mask.sum())} pos, {int(neg_mask.sum())} neg)",
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(
            {"max_prob": "max over tiles (MIL)", "mean_prob": "mean over tiles"}[key],
            fontsize=9,
        )
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle(
        "Spatial CNN transferred to DFire still images (no retraining, no recalibration)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = ROOT / "results" / "figures" / "dfire_transfer_roc.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
