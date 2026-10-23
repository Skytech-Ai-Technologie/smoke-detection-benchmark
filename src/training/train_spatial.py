"""Stage A / Stage B: train the spatial candidate classifier.

Stage A performs spatial pretraining (D-Fire, when the archive is available).
Stage B adapts the spatial model to FIRESENSE using **training videos only**.

The script never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.augment import AugmentConfig
from ..models.spatial_cnn import SpatialCNN, SpatialCNNConfig, build_spatial_cnn
from ..utils.checkpoints import load_checkpoint, save_checkpoint
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id, results_dir
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import make_generator, set_seed
from .datasets import (
    CropDataset,
    FrameCropDataset,
    VideoFeatures,
    collect_frame_groups,
    load_video_features,
    mil_collate,
)

__all__ = ["train_spatial_model", "collect_crops", "out_of_fold_logits", "main"]

LOGGER = get_logger("training.train_spatial")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def collect_crops(
    videos: Sequence[VideoFeatures],
    *,
    include_ignore: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the labelled crops of several videos.

    Args:
        videos: The videos to collect crops from.
        include_ignore: Keep the three-valued ``-1`` (ignore) samples.

    Returns:
        ``(crops, labels)``.
    """
    crop_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for video in videos:
        if video.crops.size == 0:
            continue
        labels = video.cand_labels
        keep = labels >= 0 if not include_ignore else np.ones(labels.shape, dtype=bool)
        if not keep.any():
            continue
        crop_list.append(video.crops[keep])
        label_list.append(labels[keep].astype(np.float32))
    if not crop_list:
        return np.zeros((0, 50, 50, 3), np.uint8), np.zeros(0, np.float32)
    return np.concatenate(crop_list, axis=0), np.concatenate(label_list, axis=0)


def _class_weights(labels: np.ndarray) -> torch.Tensor | None:
    positives = float((labels > 0.5).sum())
    negatives = float(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        return None
    total = positives + negatives
    return torch.tensor(
        [total / (2.0 * negatives), total / (2.0 * positives)],
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _mil_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    negative_top_k: int = 1,
) -> torch.Tensor:
    """Multiple-instance loss over the candidates of a batch of frames.

    A positive frame must contain at least one smoke-like candidate (the loss
    is applied to the maximum candidate logit); a negative frame must not
    contain any (the loss is applied to the maximum candidate logit as well,
    because the decision of the detector is taken on the maximum candidate
    probability: a negative frame whose best candidate looks like smoke
    raises an alarm, and the loss has to penalise exactly that).

    ``negative_top_k`` > 1 replaces the maximum by the mean of the k highest
    candidate logits. It is a smoother alternative, but it dilutes a single
    confident false positive and therefore weakens the penalty, which is why
    the default is the maximum.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses: list[torch.Tensor] = []
    start = 0
    for size, label in zip(group_sizes.tolist(), labels.tolist()):
        end = start + size
        chunk = logits[start:end]
        start = end
        if chunk.numel() == 0:
            continue
        if label >= 0.5:
            score = chunk.max()
            target = score.new_ones(())
            positive = criterion(score, target)
            if pos_weight is not None:
                positive = positive * pos_weight
        else:
            k = min(max(1, negative_top_k), chunk.numel())
            score = torch.topk(chunk, k).values.mean()
            target = score.new_zeros(())
            positive = criterion(score, target)
        losses.append(positive)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_spatial_model(
    train_crops: np.ndarray,
    train_labels: np.ndarray,
    val_crops: np.ndarray,
    val_labels: np.ndarray,
    *,
    model_config: SpatialCNNConfig,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weighting: bool = True,
    patience: int = 8,
    augment: bool = False,
    augment_config: AugmentConfig | None = None,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 0,
    objective: str = "crop",
    train_groups: list | None = None,
    val_groups: list | None = None,
    mil_pos_weight: float | None = None,
    negative_top_k: int = 1,
) -> tuple[SpatialCNN, dict[str, Any]]:
    """Train a candidate classifier and return the best model plus its history.

    Args:
        objective: ``crop`` for a per-crop supervised loss (used with
            region-annotated datasets such as D-Fire) or ``mil`` for the
            weakly-supervised frame-level objective (used with FIRESENSE).
        train_groups: MIL groups (crops + frame label) of the training videos.
        val_groups: MIL groups of the validation videos.
    """
    set_seed(seed)
    dev = torch.device(device)
    model = build_spatial_cnn(model_config).to(dev)

    if objective == "mil" and mil_pos_weight is None and train_groups:
        n_pos = sum(1 for _, label in train_groups if label >= 0.5)
        n_neg = len(train_groups) - n_pos
        if n_pos > 0 and n_neg > 0:
            mil_pos_weight = n_neg / n_pos
            LOGGER.info("MIL frame-class weight for the positive class: %.3f", mil_pos_weight)
    mil_pos_weight_tensor = (
        torch.tensor(float(mil_pos_weight), dtype=torch.float32) if mil_pos_weight else None
    )

    if objective == "mil":
        if not train_groups:
            raise ValueError("the MIL objective needs the training groups")
        train_ds: Dataset = FrameCropDataset(
            train_groups,
            input_size=tuple(model_config.input_size),
            augment=augment,
            augment_config=augment_config,
            seed=seed,
        )
        val_ds: Dataset = FrameCropDataset(
            val_groups or [],
            input_size=tuple(model_config.input_size),
            augment=False,
        )
        collate = mil_collate
    else:
        train_ds = CropDataset(
            train_crops,
            train_labels,
            input_size=tuple(model_config.input_size),
            augment=augment,
            augment_config=augment_config,
            seed=seed,
        )
        val_ds = CropDataset(
            val_crops,
            val_labels,
            input_size=tuple(model_config.input_size),
            augment=False,
        )
        collate = None  # type: ignore[assignment]

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=len(train_ds) > batch_size,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=_class_weights(train_labels) if class_weighting else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "val_auc": [], "epoch": []}
    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            if objective == "mil":
                images = batch["crops"].to(dev)
                labels = batch["labels"].to(dev)
                sizes = batch["group_sizes"]
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = _mil_loss(
                    logits,
                    labels,
                    sizes,
                    pos_weight=mil_pos_weight_tensor,
                    negative_top_k=negative_top_k,
                )
            else:
                images = batch[0].to(dev)
                targets = batch[1].to(dev)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_loss, val_auc = _evaluate_spatial(model, val_loader, dev, objective=objective)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if log_every and epoch % log_every == 0:
            LOGGER.info(
                "epoch %3d | train %.4f | val %.4f | val AUC %.4f",
                epoch,
                history["train_loss"][-1],
                val_loss,
                val_auc,
            )

        if not np.isnan(val_auc):
            score = val_auc
        else:
            # The validation split may contain a single class; fall back to
            # the validation loss as the model-selection criterion.
            score = -val_loss
        if score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_auc"] = float(best_auc)
    return model, history


@torch.no_grad()
def _evaluate_spatial(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: str = "crop",
) -> tuple[float, float]:
    """Validation loss and the frame-level AUROC of the candidate classifier.

    The AUROC is computed on the *frame score* (the maximum candidate
    probability) against the frame label, which is the quantity the detector
    actually uses.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    losses: list[float] = []
    frame_scores: list[float] = []
    frame_labels: list[float] = []

    if objective == "mil":
        for batch in loader:
            crops = batch["crops"].to(device)
            labels = batch["labels"].to(device)
            sizes = batch["group_sizes"]
            logits = model(crops)
            start = 0
            for size, label in zip(sizes.tolist(), labels.tolist()):
                chunk = logits[start : start + size]
                start += size
                if chunk.numel() == 0:
                    continue
                score = chunk.max()
                target = torch.ones_like(score) if label >= 0.5 else torch.zeros_like(score)
                losses.append(float(nn.functional.binary_cross_entropy_with_logits(score, target).item()))
                frame_scores.append(float(torch.sigmoid(score).item()))
                frame_labels.append(float(label))
    else:
        criterion = nn.BCEWithLogitsLoss()
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            losses.append(float(criterion(logits, targets.to(device)).item()))
            frame_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            frame_labels.extend(targets.numpy().tolist())

    if not frame_scores:
        return float("nan"), float("nan")
    y_score = np.asarray(frame_scores)
    y_true = np.asarray(frame_labels)
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    return float(np.mean(losses)), auc


def out_of_fold_logits(
    videos: Sequence[VideoFeatures],
    *,
    model_config: SpatialCNNConfig,
    n_folds: int = 3,
    seed: int = 42,
    device: str = "cpu",
    objective: str = "mil",
    return_histories: bool = False,
    **train_kwargs,
):
    """Produce *out-of-fold* spatial logits for the training videos.

    Training the temporal model on features produced by a spatial model that
    has already seen the same frames makes the frame probabilities
    optimistically confident. Cross-fitting removes that bias: every training
    video is scored by a spatial model that did not see it.

    The per-fold training histories are returned as well: they are used to
    select the number of epochs of the final model without touching the
    validation or the test split.
    """
    if len(videos) < n_folds:
        n_folds = max(1, len(videos))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    logits: dict[str, np.ndarray] = {}
    histories: list[dict[str, Any]] = []

    groups_by_video = {
        video.video_id: collect_frame_groups([video]) for video in videos
    }

    for fold in range(n_folds):
        val_idx = {int(i) for i in order[fold::n_folds]}
        train_videos = [v for i, v in enumerate(videos) if i not in val_idx]
        val_videos = [v for i, v in enumerate(videos) if i in val_idx]
        if not train_videos or not val_videos:
            continue

        train_groups = [g for v in train_videos for g in groups_by_video[v.video_id]]
        val_groups = [g for v in val_videos for g in groups_by_video[v.video_id]]
        if not train_groups:
            continue

        model, history = train_spatial_model(
            np.zeros((0, 50, 50, 3), np.uint8),
            np.zeros(0, np.float32),
            np.zeros((0, 50, 50, 3), np.uint8),
            np.zeros(0, np.float32),
            model_config=model_config,
            seed=seed + fold,
            device=device,
            objective=objective,
            train_groups=train_groups,
            val_groups=val_groups,
            **train_kwargs,
        )
        histories.append(history)

        from .datasets import score_candidates

        for video in val_videos:
            logits[video.video_id] = score_candidates(model, video, device=device)
        LOGGER.info("out-of-fold fold %d/%d done", fold + 1, n_folds)

    if return_histories:
        return logits, histories
    return logits


def select_epoch_count(histories: Sequence[dict[str, Any]], criterion: str = "val_loss") -> int:
    """Select the number of epochs by cross-validation inside the training split.

    The official validation split is *not* used for this decision, because
    it can contain a single class (a negative video without any candidate),
    which makes the validation loss uninformative.
    """
    if not histories:
        return 10
    length = min(len(h.get(criterion, [])) for h in histories)
    if length == 0:
        return 10
    curves = np.asarray([h[criterion][:length] for h in histories], dtype=np.float64)
    mean_curve = np.nanmean(curves, axis=0)
    return int(np.nanargmin(mean_curve)) + 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_split_roles(splits_file: str | Path, split: str) -> dict[str, list[str]]:
    from ..data.video_split import read_manifest

    rows = read_manifest(splits_file)
    roles: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        if row.get("split_id") != split:
            continue
        roles.setdefault(row["role"], []).append(row["video_id"])
    return roles


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the spatial candidate classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None, help="override the dataset name")
    parser.add_argument("--checkpoint", default=None, help="initialise from this checkpoint")
    parser.add_argument("--split", default=None, help="split id, e.g. split01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="checkpoint output path")
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--no-cv", action="store_true", help="skip the cross-validated epoch selection")
    parser.add_argument("--epochs", type=int, default=None, help="override the number of epochs")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = SpatialCNNConfig.from_dict(config.get("model", {}))

    dataset = args.dataset or data_cfg.get("dataset", "firesense")
    split = args.split or data_cfg.get("split", "split01")
    feature_dir = Path(data_cfg.get("feature_dir", "data/processed/features"))
    splits_file = Path(data_cfg.get("splits_file", "data/manifests/splits.csv"))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    roles = _load_split_roles(splits_file, split)
    train_ids = set(roles.get("train", []))
    val_ids = set(roles.get("validation", []))

    def load_ids(ids: set[str]) -> list[VideoFeatures]:
        videos: list[VideoFeatures] = []
        for path in sorted(feature_dir.glob("*.npz")):
            if path.stem in ids:
                videos.append(load_video_features(path))
        return videos

    train_videos = load_ids(train_ids)
    val_videos = load_ids(val_ids)
    if not train_videos:
        raise RuntimeError(f"no training videos found in {feature_dir} for split {split}")
    LOGGER.info(
        "split=%s | train videos=%d | validation videos=%d",
        split,
        len(train_videos),
        len(val_videos),
    )

    train_crops, train_labels = collect_crops(train_videos)
    val_crops, val_labels = collect_crops(val_videos) if val_videos else (np.zeros((0, 50, 50, 3), np.uint8), np.zeros(0, np.float32))
    LOGGER.info(
        "crops: train=%d (pos=%d) | val=%d (pos=%d)",
        train_labels.size,
        int(train_labels.sum()),
        val_labels.size,
        int(val_labels.sum()),
    )

    # FIRESENSE has no region-level annotation, so the spatial classifier is
    # trained with the weakly-supervised (MIL) frame-level objective.
    objective = str(train_cfg.get("objective", "mil" if dataset == "firesense" else "crop"))
    train_groups = collect_frame_groups(train_videos)
    val_groups = collect_frame_groups(val_videos) if val_videos else []
    if objective == "mil":
        pos_frames = sum(1 for _, label in train_groups if label >= 0.5)
        val_pos = sum(1 for _, label in val_groups if label >= 0.5)
        LOGGER.info(
            "MIL objective | train groups=%d (positive frames=%d) | val groups=%d (positive frames=%d)",
            len(train_groups),
            pos_frames,
            len(val_groups),
            val_pos,
        )
        for video in val_videos:
            LOGGER.info(
                "  val video %s (%s): %d candidate frames",
                video.video_id,
                video.label,
                int((video.cand_frame_index >= 0).sum()) and len(set(video.cand_frame_index.tolist())),
            )

    model = build_spatial_cnn(model_cfg)
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.model)
        LOGGER.info("initialised from %s", args.checkpoint)

    epochs = int(train_cfg.get("epochs", 40))
    patience = int(train_cfg.get("early_stopping_patience", 8))
    if args.epochs is not None:
        epochs = int(args.epochs)
    if args.no_cv:
        LOGGER.info("cross-validated epoch selection disabled by --no-cv")
        patience = 10**6

    # The number of epochs is selected by cross-validation *inside* the
    # training split, so that the official validation split is never used to
    # fit the spatial model. This also produces the out-of-fold spatial
    # logits that the temporal training needs.
    if objective == "mil" and bool(train_cfg.get("select_epochs_by_cv", True)) and not args.no_cv:
        LOGGER.info("cross-validating the spatial model inside the training split")
        _, histories = out_of_fold_logits(
            train_videos,
            model_config=model_cfg,
            n_folds=int(train_cfg.get("out_of_fold_folds", 3)),
            seed=args.seed,
            device=device,
            objective=objective,
            return_histories=True,
            epochs=epochs,
            patience=10**6,  # no early stopping: the full curve is needed
            batch_size=int(train_cfg.get("batch_size", 64)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            class_weighting=bool(train_cfg.get("class_weighting", True)),
            augment=bool(train_cfg.get("augment", False)),
            augment_config=AugmentConfig.from_dict(train_cfg.get("augment_config")),
            log_every=0,
        )
        epochs = select_epoch_count(histories)
        patience = 10**6
        LOGGER.info("selected %d epochs by cross-validation", epochs)

    model, history = train_spatial_model(
        train_crops,
        train_labels,
        val_crops,
        val_labels,
        model_config=model_cfg,
        epochs=epochs,
        batch_size=int(train_cfg.get("batch_size", 64)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        class_weighting=bool(train_cfg.get("class_weighting", True)),
        patience=patience,
        augment=bool(train_cfg.get("augment", False)),
        augment_config=AugmentConfig.from_dict(train_cfg.get("augment_config")),
        seed=args.seed,
        device=device,
        log_every=int(train_cfg.get("log_every", 5)),
        objective=objective,
        train_groups=train_groups,
        val_groups=val_groups,
        negative_top_k=int(train_cfg.get("negative_top_k", 1)),
    )

    exp_id = experiment_id("spatial", split=split, seed=args.seed, extra=[dataset])
    out_dir = results_dir(args.out_dir, exp_id)
    checkpoint_path = Path(args.out) if args.out else (out_dir / "model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        checkpoint_path,
        model,
        meta={
            "model_config": model_cfg.to_dict(),
            "split": split,
            "seed": args.seed,
            "dataset": dataset,
            "history": history,
            "config": config,
        },
    )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_history(history, out_dir / "training_curves.png")
    confusion = _confusion_matrix(
        val_groups if objective == "mil" else None,
        val_crops,
        val_labels,
        model,
        device,
    )
    (out_dir / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    _plot_confusion(confusion, out_dir / "confusion_matrix.png")

    record = build_record(
        exp_id=exp_id,
        config=config,
        manifest_path=splits_file,
        seed=args.seed,
        device=device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=checkpoint_path,
        split=split,
        model=f"spatial_cnn/{dataset}",
        notes=f"stage={'A' if dataset == 'dfire' else 'B'}",
        metrics={
            "best_val_auc": history.get("best_val_auc"),
            "best_epoch": history.get("best_epoch"),
            "n_train_crops": int(train_labels.size),
            "n_val_crops": int(val_labels.size),
        },
    )
    append_to_registry(args.registry, record)
    (out_dir / "experiment.json").write_text(record.to_json(), encoding="utf-8")
    LOGGER.info("saved experiment to %s", out_dir)
    return 0


@torch.no_grad()
def _confusion_matrix(
    groups: list | None,
    crops: np.ndarray,
    labels: np.ndarray,
    model: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Confusion matrix of the frame-level decision (MIL) or of the crops."""
    from ..models.spatial_cnn import preprocess_crops

    model.eval()
    scores: list[float] = []
    truth: list[int] = []

    if groups is not None:
        for group_crops, label in groups:
            if group_crops.shape[0] == 0:
                continue
            tensor = preprocess_crops(group_crops).to(device)
            probs = torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1)
            scores.append(float(probs.max()))
            truth.append(int(label >= 0.5))
    else:
        if labels.size == 0:
            return {"threshold": threshold, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for start in range(0, crops.shape[0], 256):
            batch = crops[start : start + 256]
            tensor = preprocess_crops(batch).to(device)
            scores.extend(torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1).tolist())
            truth.extend(labels[start : start + 256].astype(int).tolist())

    if not scores:
        return {"threshold": threshold, "tn": 0, "fp": 0, "fn": 0, "tp": 0}

    pred = (np.asarray(scores) >= threshold).astype(int)
    y = np.asarray(truth, dtype=int)
    return {
        "threshold": threshold,
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def _plot_history(history: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = history.get("epoch", [])
    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].set_title("Spatial training loss")
    axes[1].plot(epochs, history.get("val_auc", []), color="C1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation AUROC")
    axes[1].set_title("Spatial validation AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(confusion: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    matrix = np.array(
        [[confusion.get("tn", 0), confusion.get("fp", 0)], [confusion.get("fn", 0), confusion.get("tp", 0)]],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["background", "smoke"])
    ax.set_yticks([0, 1], ["background", "smoke"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
