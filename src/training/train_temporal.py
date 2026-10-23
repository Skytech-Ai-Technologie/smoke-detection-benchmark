"""Stage C: train the temporal models and fit the evidence accumulator.

Three kinds of temporal module are supported through the same entry point:

* ``gru`` / ``tcn`` -- neural models trained with masked binary
  cross-entropy on the *current* and *horizon* heads,
* ``evidence`` -- the non-neural probabilistic accumulator; its reliability
  model is fitted on the training split and its thresholds are selected on
  the validation split,
* ``voting`` -- the fixed temporal rules, which have no trainable
  parameters; only their thresholds are selected on the validation split.

The spatial model that produces the frame probabilities is always a model
trained on the *training* split only. For the training videos the logits are
produced **out of fold** (cross-fitting), so that the temporal model is not
trained on optimistically confident frame probabilities.
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
from torch.utils.data import DataLoader

from ..data.video_split import read_manifest
from ..models.candidate_features import FEATURE_DIM, FeatureScaler
from ..models.evidence_accumulator import (
    EvidenceAccumulator,
    EvidenceConfig,
    ReliabilityModel,
    reliability_features,
)
from ..models.fixed_temporal import TemporalVoter, VotingConfig
from ..models.spatial_cnn import SpatialCNN, SpatialCNNConfig, build_spatial_cnn
from ..models.temporal_gru import GRUConfig, TemporalGRU
from ..models.temporal_tcn import TCNConfig, TemporalTCN
from ..utils.checkpoints import load_checkpoint, save_checkpoint, sha256_file
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id, results_dir
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import make_generator, set_seed
from .datasets import (
    VideoFeatures,
    build_temporal_dataset,
    frame_features_for_video,
    load_video_features,
    score_candidates,
)

__all__ = ["prepare_features", "train_neural_temporal", "fit_evidence_accumulator", "main"]

LOGGER = get_logger("training.train_temporal")


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #
def _load_split_roles(splits_file: str | Path, split: str) -> dict[str, list[str]]:
    rows = read_manifest(splits_file)
    roles: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        if row.get("split_id") != split:
            continue
        roles.setdefault(row["role"], []).append(row["video_id"])
    return roles


def prepare_features(
    videos: Sequence[VideoFeatures],
    spatial_model: SpatialCNN,
    *,
    device: str = "cpu",
    out_of_fold_logits: dict[str, np.ndarray] | None = None,
    max_candidates: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the per-frame feature matrix and the per-candidate logits.

    Args:
        videos: The videos to featurise.
        spatial_model: The spatial model used to score the candidate crops.
        device: Torch device.
        out_of_fold_logits: Optional pre-computed logits (cross-fitted) that
            take precedence over the model's own predictions.

    Returns:
        ``(feature_matrices, candidate_logits)`` keyed by video id.
    """
    features: dict[str, np.ndarray] = {}
    logits_by_video: dict[str, np.ndarray] = {}
    for video in videos:
        if out_of_fold_logits is not None and video.video_id in out_of_fold_logits:
            logits = out_of_fold_logits[video.video_id]
        else:
            logits = score_candidates(spatial_model, video, device=device)
        logits_by_video[video.video_id] = logits
        features[video.video_id] = frame_features_for_video(
            video, logits, max_candidates=max_candidates
        )
    return features, logits_by_video


def frame_probabilities(feature_matrix: np.ndarray) -> np.ndarray:
    """Per-frame spatial probability implied by a feature matrix.

    The feature builder stores the strongest spatial logit of the frame in the
    ``cnn_prob_max`` column; this helper turns it into a probability.
    """
    from ..models.candidate_features import FEATURE_INDEX

    if feature_matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    logits = feature_matrix[:, FEATURE_INDEX["cnn_prob_max"]]
    return _sigmoid(logits)


# --------------------------------------------------------------------------- #
# Neural temporal training
# --------------------------------------------------------------------------- #
def train_neural_temporal(
    train_dataset,
    val_dataset,
    scaler: FeatureScaler,
    *,
    model: nn.Module,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    horizon_weight: float = 0.5,
    uncertainty_weight: float = 0.05,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 5,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a GRU/TCN with masked BCE on the current and horizon heads."""
    set_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "val_now_loss": [], "epoch": []}
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    without_improvement = 0

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            sequences = batch["sequence"].to(dev)
            y_now = batch["y_now"].to(dev)
            y_horizon = batch["y_horizon"].to(dev)
            m_now = batch["mask_now"].to(dev)
            m_horizon = batch["mask_horizon"].to(dev)

            if scaler.fitted:
                mean = torch.as_tensor(scaler.mean, device=dev)
                std = torch.as_tensor(scaler.std, device=dev)
                sequences = (sequences - mean) / std

            optimizer.zero_grad(set_to_none=True)
            out = model(sequences)
            loss_now = (bce(out["now"], y_now) * m_now).sum() / m_now.sum().clamp(min=1.0)
            loss_horizon = (bce(out["horizon"], y_horizon) * m_horizon).sum() / m_horizon.sum().clamp(min=1.0)
            # The uncertainty head is regularised towards a small log-variance.
            log_var = out["log_variance"].clamp(-8.0, 4.0)
            loss_unc = (log_var**2).mean()
            loss = loss_now + horizon_weight * loss_horizon + uncertainty_weight * loss_unc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_loss, val_now = _evaluate_temporal(model, val_loader, dev, scaler)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_loss"].append(val_loss)
        history["val_now_loss"].append(val_now)

        if log_every and epoch % log_every == 0:
            LOGGER.info(
                "epoch %3d | train %.4f | val %.4f | val(now) %.4f",
                epoch,
                history["train_loss"][-1],
                val_loss,
                val_now,
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= patience:
                LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_loss)
    return model, history


@torch.no_grad()
def _evaluate_temporal(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: FeatureScaler,
) -> tuple[float, float]:
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.eval()
    losses: list[float] = []
    now_losses: list[float] = []
    for batch in loader:
        sequences = batch["sequence"].to(device)
        y_now = batch["y_now"].to(device)
        y_horizon = batch["y_horizon"].to(device)
        m_now = batch["mask_now"].to(device)
        m_horizon = batch["mask_horizon"].to(device)
        if scaler.fitted:
            mean = torch.as_tensor(scaler.mean, device=device)
            std = torch.as_tensor(scaler.std, device=device)
            sequences = (sequences - mean) / std
        out = model(sequences)
        loss_now = (bce(out["now"], y_now) * m_now).sum() / m_now.sum().clamp(min=1.0)
        loss_horizon = (bce(out["horizon"], y_horizon) * m_horizon).sum() / m_horizon.sum().clamp(min=1.0)
        losses.append(float((loss_now + 0.5 * loss_horizon).item()))
        now_losses.append(float(loss_now.item()))
    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(now_losses))


# --------------------------------------------------------------------------- #
# Evidence accumulator
# --------------------------------------------------------------------------- #
def _run_evidence_stream(
    video: VideoFeatures,
    frame_probs: np.ndarray,
    reliability: np.ndarray,
    config: EvidenceConfig,
) -> np.ndarray:
    """Return the per-frame alarm mask produced by the evidence accumulator."""
    accumulator = EvidenceAccumulator(config, ReliabilityModel())
    mask = np.zeros(len(frame_probs), dtype=bool)
    for t in range(len(frame_probs)):
        out = accumulator.step(float(frame_probs[t]), reliability=float(reliability[t]))
        mask[t] = bool(out["alarm"])
    return mask


def fit_evidence_accumulator(
    train_videos: Sequence[VideoFeatures],
    train_features: dict[str, np.ndarray],
    val_videos: Sequence[VideoFeatures],
    val_features: dict[str, np.ndarray],
    *,
    base_config: EvidenceConfig,
    false_alarm_penalty: float = 1.0,
    alarm_grid: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    cancel_grid: Sequence[float] = (-8.0, -6.0, -4.0, -2.0, 0.0),
    forgetting: float = 0.95,
) -> tuple[EvidenceConfig, ReliabilityModel, dict[str, Any]]:
    """Fit the reliability model on the training split and select the thresholds.

    The thresholds are chosen on the **validation** split by maximising
    ``video_recall - penalty * false_alarms_per_hour``.
    """
    from ..evaluation.metrics import aggregate_video_metrics, evaluate_video_events

    # --- reliability model (fitted on the training split) ------------------
    rel_features: list[np.ndarray] = []
    rel_labels: list[int] = []
    for video in train_videos:
        matrix = train_features.get(video.video_id)
        if matrix is None or matrix.size == 0:
            continue
        rel_features.append(_reliability_matrix(matrix))
        # A frame whose spatial probability is high is a reliable observation.
        probs = frame_probabilities(matrix)
        rel_labels.extend((probs >= 0.5).astype(int).tolist())
    reliability_model = ReliabilityModel()
    if rel_features:
        reliability_model.fit(np.concatenate(rel_features, axis=0), np.asarray(rel_labels))

    # --- threshold selection on the validation split ------------------------
    best: tuple[float, EvidenceConfig, dict[str, Any]] | None = None
    for alarm in alarm_grid:
        for cancel in cancel_grid:
            if cancel >= alarm:
                continue
            config = EvidenceConfig(
                alarm_threshold=float(alarm),
                cancel_threshold=float(cancel),
                forgetting_factor=forgetting,
                min_reliability=base_config.min_reliability,
                max_reliability=base_config.max_reliability,
                clip_logit=base_config.clip_logit,
                use_reliability_model=base_config.use_reliability_model,
            )
            evaluations = []
            for video in val_videos:
                matrix = val_features.get(video.video_id)
                if matrix is None:
                    continue
                probs = frame_probabilities(matrix)
                reliability = _reliability_values(matrix, reliability_model)
                mask = _run_evidence_stream(video, probs, reliability, config)
                evaluations.append(
                    evaluate_video_events(
                        video.video_id,
                        video.label,
                        video.fps,
                        mask,
                        video.onset_frame,
                    )
                )
            metrics = aggregate_video_metrics(evaluations)
            score = metrics["video_recall"] - false_alarm_penalty * metrics["false_alarms_per_hour"]
            if best is None or score > best[0]:
                best = (score, config, metrics)

    assert best is not None
    _, config, metrics = best
    LOGGER.info(
        "evidence thresholds selected: alarm=%.1f cancel=%.1f | recall=%.3f fa/h=%.2f",
        config.alarm_threshold,
        config.cancel_threshold,
        metrics["video_recall"],
        metrics["false_alarms_per_hour"],
    )
    return config, reliability_model, metrics


def _reliability_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Reliability features of every frame, derived from the frame features."""
    n = feature_matrix.shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    for t in range(n):
        out[t] = reliability_features(feature_matrix[t], uncertainty=0.0, motion_consistency=0.0)
    return out


def _reliability_values(
    feature_matrix: np.ndarray,
    reliability_model: ReliabilityModel,
) -> np.ndarray:
    feats = _reliability_matrix(feature_matrix)
    if not reliability_model.fitted or feats.size == 0:
        return np.ones(feature_matrix.shape[0], dtype=np.float64)
    return reliability_model.predict(feats)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train a temporal model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--no-out-of-fold", action="store_true", help="disable cross-fitting")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    kind = str(model_cfg.get("type", "tcn")).lower()
    split = args.split or data_cfg.get("split", "split01")
    feature_dir = Path(data_cfg.get("feature_dir", "data/processed/features"))
    splits_file = Path(data_cfg.get("splits_file", "data/manifests/splits.csv"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    spatial_ckpt = load_checkpoint(args.spatial_checkpoint, map_location="cpu")
    spatial_meta = spatial_ckpt.meta
    spatial_model = build_spatial_cnn(spatial_meta.get("model_config", {}))
    spatial_model.load_state_dict(spatial_ckpt.model)
    spatial_model.to(device).eval()
    LOGGER.info("spatial checkpoint: %s (sha256=%s)", args.spatial_checkpoint, spatial_ckpt.sha256[:16])

    roles = _load_split_roles(splits_file, split)
    ids_by_role = {role: set(roles.get(role, [])) for role in ("train", "validation", "test")}

    def load_videos(ids: set[str]) -> list[VideoFeatures]:
        out: list[VideoFeatures] = []
        for path in sorted(feature_dir.glob("*.npz")):
            if path.stem in ids:
                out.append(load_video_features(path))
        return out

    train_videos = load_videos(ids_by_role["train"])
    val_videos = load_videos(ids_by_role["validation"])
    if not train_videos:
        raise RuntimeError(f"no training videos in {feature_dir} for {split}")

    sequence_length = int(model_cfg.get("sequence_length", 16))
    horizon = int(model_cfg.get("horizon", 8))
    guard_frames = int(data_cfg.get("pre_onset_guard_frames", 0))

    out_dir = results_dir(args.out_dir, experiment_id(kind, split=split, seed=args.seed))
    metrics: dict[str, Any] = {}

    # ---------------------------------------------------------------- evidence
    if kind == "evidence":
        from .train_spatial import out_of_fold_logits

        oof = None
        if not args.no_out_of_fold and train_cfg.get("use_out_of_fold", True):
            oof = out_of_fold_logits(
                train_videos,
                model_config=SpatialCNNConfig.from_dict(spatial_meta.get("model_config", {})),
                n_folds=int(train_cfg.get("out_of_fold_folds", 3)),
                seed=args.seed,
                device=device,
                epochs=int(train_cfg.get("spatial_epochs", 30)),
                batch_size=int(train_cfg.get("spatial_batch_size", 64)),
            )
        train_features, _ = prepare_features(
            train_videos, spatial_model, device=device, out_of_fold_logits=oof
        )
        val_features, _ = prepare_features(val_videos, spatial_model, device=device)

        base = EvidenceConfig.from_dict(config.get("evidence", {}))
        ev_config, reliability_model, metrics = fit_evidence_accumulator(
            train_videos,
            train_features,
            val_videos,
            val_features,
            base_config=base,
            false_alarm_penalty=float(config.get("false_alarm_penalty", 1.0)),
            forgetting=float(config.get("forgetting_factor", 0.95)),
        )
        payload = {
            "evidence_config": ev_config.to_dict(),
            "reliability_model": reliability_model.state_dict(),
            "spatial_checkpoint": args.spatial_checkpoint,
            "spatial_checkpoint_sha256": spatial_ckpt.sha256,
            "split": split,
            "seed": args.seed,
        }
        checkpoint_path = Path(args.out) if args.out else (out_dir / "evidence.pt")
        torch.save(payload, checkpoint_path)
        (out_dir / "evidence_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- voting
    elif kind == "voting":
        voting_cfg = VotingConfig.from_dict(config.get("voting", {}))
        payload = {
            "voting_config": voting_cfg.to_dict(),
            "spatial_checkpoint": args.spatial_checkpoint,
            "spatial_checkpoint_sha256": spatial_ckpt.sha256,
            "split": split,
            "seed": args.seed,
        }
        checkpoint_path = Path(args.out) if args.out else (out_dir / "voting.pt")
        torch.save(payload, checkpoint_path)
        (out_dir / "voting_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -------------------------------------------------------- neural (gru/tcn)
    else:
        from .train_spatial import out_of_fold_logits

        oof = None
        if not args.no_out_of_fold and train_cfg.get("use_out_of_fold", True):
            LOGGER.info("computing out-of-fold spatial logits for the training videos")
            oof = out_of_fold_logits(
                train_videos,
                model_config=SpatialCNNConfig.from_dict(spatial_meta.get("model_config", {})),
                n_folds=int(train_cfg.get("out_of_fold_folds", 3)),
                seed=args.seed,
                device=device,
                epochs=int(train_cfg.get("spatial_epochs", 30)),
                batch_size=int(train_cfg.get("spatial_batch_size", 64)),
            )

        train_features, _ = prepare_features(
            train_videos, spatial_model, device=device, out_of_fold_logits=oof
        )
        val_features, _ = prepare_features(val_videos, spatial_model, device=device)

        train_dataset = build_temporal_dataset(
            train_videos,
            train_features,
            sequence_length=sequence_length,
            horizon=horizon,
            guard_frames=guard_frames,
            stride=int(data_cfg.get("sequence_stride", 1)),
        )
        val_dataset = build_temporal_dataset(
            val_videos,
            val_features,
            sequence_length=sequence_length,
            horizon=horizon,
            guard_frames=guard_frames,
            stride=int(data_cfg.get("sequence_stride", 1)),
        )
        LOGGER.info("temporal windows: train=%d val=%d", len(train_dataset), len(val_dataset))

        scaler = FeatureScaler()
        if len(train_dataset):
            scaler.fit(train_dataset.sequences.reshape(-1, train_dataset.sequences.shape[-1]))

        if kind == "gru":
            temporal = TemporalGRU(GRUConfig.from_dict(model_cfg))
        else:
            temporal = TemporalTCN(TCNConfig.from_dict(model_cfg))

        temporal, history = train_neural_temporal(
            train_dataset,
            val_dataset,
            scaler,
            model=temporal,
            epochs=int(train_cfg.get("epochs", 60)),
            batch_size=int(train_cfg.get("batch_size", 128)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            patience=int(train_cfg.get("early_stopping_patience", 10)),
            horizon_weight=float(train_cfg.get("horizon_weight", 0.5)),
            uncertainty_weight=float(train_cfg.get("uncertainty_weight", 0.05)),
            seed=args.seed,
            device=device,
            log_every=int(train_cfg.get("log_every", 5)),
        )

        checkpoint_path = Path(args.out) if args.out else (out_dir / "model.pt")
        save_checkpoint(
            checkpoint_path,
            temporal,
            meta={
                "model_config": model_cfg,
                "model_type": kind,
                "feature_scaler": scaler.state_dict(),
                "spatial_checkpoint": args.spatial_checkpoint,
                "spatial_checkpoint_sha256": spatial_ckpt.sha256,
                "split": split,
                "seed": args.seed,
                "history": history,
                "config": config,
                "sequence_length": sequence_length,
                "horizon": horizon,
            },
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        metrics = {
            "best_val_loss": history.get("best_val_loss"),
            "best_epoch": history.get("best_epoch"),
            "n_train_windows": len(train_dataset),
            "n_val_windows": len(val_dataset),
        }

    record = build_record(
        exp_id=experiment_id(kind, split=split, seed=args.seed),
        config=config,
        manifest_path=splits_file,
        seed=args.seed,
        device=device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=checkpoint_path,
        split=split,
        model=kind,
        notes="stage C",
        metrics=metrics,
    )
    append_to_registry(args.registry, record)
    (out_dir / "experiment.json").write_text(record.to_json(), encoding="utf-8")
    LOGGER.info("saved experiment to %s", out_dir)
    return 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
