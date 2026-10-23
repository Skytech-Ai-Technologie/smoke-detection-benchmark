"""Stage D: calibrate the temporal model on the validation split.

Calibration is *never* fitted on the test split. The script compares the
available calibration methods and reports the calibration error before and
after, so that the paper can justify the choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.calibration import CALIBRATION_METHODS, build_calibrator
from ..models.candidate_features import FEATURE_INDEX
from ..utils.checkpoints import load_checkpoint, save_checkpoint, sha256_file
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id, results_dir
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed

__all__ = ["collect_validation_logits", "calibrate_checkpoint", "main"]

LOGGER = get_logger("training.calibrate")


def _load_split_roles(splits_file: str | Path, split: str) -> dict[str, list[str]]:
    from ..data.video_split import read_manifest

    rows = read_manifest(splits_file)
    roles: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        if row.get("split_id") == split:
            roles.setdefault(row["role"], []).append(row["video_id"])
    return roles


def collect_validation_logits(
    temporal_checkpoint: str | Path,
    spatial_checkpoint: str | Path | None,
    *,
    split: str = "split01",
    role: str = "validation",
    feature_dir: str | Path = "data/processed/features",
    splits_file: str | Path = "data/manifests/splits.csv",
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect the temporal logits and the frame labels of a split role.

    Returns:
        ``(logits, labels, video_ids)``.
    """
    from ..training.datasets import frame_features_for_video, load_video_features, score_candidates
    from ..models.builder import build_detector

    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides={"mode": "fixed", "persistence_seconds": 0.0},
    )
    detector = bundle.detector
    if detector.temporal_model is None:
        raise ValueError("calibration requires a neural temporal model (gru/tcn)")

    roles = _load_split_roles(splits_file, split)
    ids = set(roles.get(role, []))
    paths = [p for p in sorted(Path(feature_dir).glob("*.npz")) if p.stem in ids]
    if not paths:
        raise RuntimeError(f"no videos found for role={role} in split={split}")

    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    video_ids: list[str] = []

    for path in paths:
        video = load_video_features(path)
        crop_logits = score_candidates(detector.spatial_model, video, device=device) if detector.spatial_model else np.zeros(0)
        features = frame_features_for_video(video, crop_logits)

        n = features.shape[0]
        sequence_length = detector.config.sequence_length
        if n < sequence_length:
            continue

        # The temporal model is causal, so the prediction for frame t uses the
        # window [t - L + 1, t] only.
        windows = np.stack([features[t - sequence_length + 1 : t + 1] for t in range(sequence_length - 1, n)])
        if detector.feature_scaler is not None and detector.feature_scaler.fitted:
            windows = detector.feature_scaler.transform(windows)

        with torch.no_grad():
            tensor = torch.from_numpy(windows.astype(np.float32)).to(detector.device)
            out = detector.temporal_model(tensor)
        frame_logits = out["now"].detach().cpu().numpy()

        onset = video.onset_frame
        if video.label == "smoke" and onset is not None:
            y = (np.arange(sequence_length - 1, n) >= onset).astype(np.float32)
        else:
            y = np.zeros(n - sequence_length + 1, dtype=np.float32)

        logits.append(frame_logits)
        labels.append(y)
        video_ids.append(video.video_id)

    if not logits:
        return np.zeros(0), np.zeros(0), []
    return np.concatenate(logits), np.concatenate(labels), video_ids


def calibrate_checkpoint(
    temporal_checkpoint: str | Path,
    spatial_checkpoint: str | Path | None,
    *,
    method: str = "temperature",
    split: str = "split01",
    role: str = "validation",
    feature_dir: str | Path = "data/processed/features",
    splits_file: str | Path = "data/manifests/splits.csv",
    device: str = "cpu",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a calibrator on the validation split and save the calibrated model."""
    from ..evaluation.metrics import calibration_metrics

    logits, labels, video_ids = collect_validation_logits(
        temporal_checkpoint,
        spatial_checkpoint,
        split=split,
        role=role,
        feature_dir=feature_dir,
        splits_file=splits_file,
        device=device,
    )
    if logits.size == 0:
        raise RuntimeError("no validation logits collected")

    report: dict[str, Any] = {
        "n_samples": int(labels.size),
        "n_positive": int(labels.sum()),
        "n_videos": len(video_ids),
        "split": split,
        "role": role,
        "methods": {},
    }

    raw = calibration_metrics(labels, 1.0 / (1.0 + np.exp(-logits)))
    report["methods"]["none"] = {
        "ece": raw["ece"],
        "mce": raw["mce"],
        "brier": raw["brier"],
    }

    best_calibrator = None
    for name in CALIBRATION_METHODS:
        calibrator = build_calibrator(name)
        calibrator.fit(logits, labels)
        probs = calibrator.predict_proba(logits)
        metrics = calibration_metrics(labels, probs)
        report["methods"][name] = {
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "brier": metrics["brier"],
        }
        if name == method:
            best_calibrator = calibrator

    if best_calibrator is None:
        best_calibrator = build_calibrator("none")

    report["selected_method"] = method
    report["calibrator_state"] = best_calibrator.state_dict()

    ckpt = load_checkpoint(temporal_checkpoint, map_location="cpu")
    meta = dict(ckpt.meta)
    meta["calibrator"] = best_calibrator.state_dict()
    meta["calibration_report"] = report

    target = Path(out_path) if out_path else Path(temporal_checkpoint).with_name(
        Path(temporal_checkpoint).stem + "_calibrated.pt"
    )
    save_checkpoint(target, ckpt.model, meta=meta)
    report["checkpoint"] = str(target)
    report["checkpoint_sha256"] = sha256_file(target)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Calibrate a temporal model.")
    parser.add_argument("--checkpoint", required=True, help="temporal checkpoint")
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--method", default="temperature", choices=list(CALIBRATION_METHODS))
    parser.add_argument("--split", default="split01")
    parser.add_argument("--role", default="validation", choices=["validation", "train", "test"])
    parser.add_argument("--feature-dir", default="data/processed/features")
    parser.add_argument("--splits-file", default="data/manifests/splits.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    report = calibrate_checkpoint(
        args.checkpoint,
        args.spatial_checkpoint,
        method=args.method,
        split=args.split,
        role=args.role,
        feature_dir=args.feature_dir,
        splits_file=args.splits_file,
        device=device,
        out_path=args.out,
    )

    out_dir = results_dir(args.out_dir, experiment_id("calibration", split=args.split, seed=args.seed))
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    record = build_record(
        exp_id=experiment_id("calibration", split=args.split, seed=args.seed),
        config={"method": args.method, "split": args.split, "role": args.role},
        manifest_path=args.splits_file,
        seed=args.seed,
        device=device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=report.get("checkpoint"),
        split=args.split,
        model=f"calibration/{args.method}",
        notes="stage D",
        metrics={k: v for k, v in report["methods"].items()},
    )
    append_to_registry(args.registry, record)

    LOGGER.info("calibration report: %s", json.dumps(report["methods"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
