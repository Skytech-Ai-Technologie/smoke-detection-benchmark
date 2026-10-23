"""Shared command-line plumbing for the evaluation entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger

__all__ = [
    "add_common_arguments",
    "resolve_config",
    "load_benchmark_config",
    "write_result",
    "summarise",
]

LOGGER = get_logger("evaluation.common")

DEFAULT_MANIFEST = "data/manifests/firesense_videos.csv"
DEFAULT_SPLITS = "data/manifests/splits.csv"
DEFAULT_FEATURES = "data/processed/features"


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments shared by every evaluation entry point."""
    parser.add_argument("--config", default=None, help="benchmark configuration YAML")
    parser.add_argument("--model", default=None, help="temporal checkpoint (or 'none')")
    parser.add_argument("--spatial-checkpoint", default=None, help="spatial CNN checkpoint")
    parser.add_argument("--manifest", default=None, help="dataset manifest CSV")
    parser.add_argument("--feature-dir", default=None, help="cached feature directory")
    parser.add_argument("--splits-file", default=None, help="split registry CSV")
    parser.add_argument("--split", default=None, help="split id, e.g. split01")
    parser.add_argument("--role", default=None, choices=["train", "validation", "test"])
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--score-source", default=None, choices=["temporal", "spatial"])
    parser.add_argument("--policy-mode", default=None, choices=["fixed", "adaptive"])
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--persistence-seconds", type=float, default=None)
    parser.add_argument("--decision-window-seconds", type=float, default=None)
    parser.add_argument("--refractory-seconds", type=float, default=None)
    parser.add_argument("--frame-skip", action="store_true", default=None)
    parser.add_argument("--no-frame-skip", dest="frame_skip", action="store_false")
    parser.add_argument(
        "--policy-file",
        default=None,
        help="JSON produced by src.training.select_thresholds; the selected operating point "
        "is applied to the evaluation unchanged",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="reference model variant: spatial, voting_mean, voting_median, "
        "voting_majority, voting_max or voting_persistence",
    )
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--tag", default=None, help="name suffix for the output files")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_benchmark_config(path: str | Path | None) -> dict[str, Any]:
    """Load the benchmark configuration (empty dict when no file is given)."""
    if not path:
        return {}
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the CLI arguments with the benchmark configuration."""
    config = load_benchmark_config(getattr(args, "config", None))
    benchmark = config.get("benchmark", {})
    data = config.get("data", {})

    args.manifest = getattr(args, "manifest", None) or data.get("manifest", DEFAULT_MANIFEST)
    args.feature_dir = getattr(args, "feature_dir", None) or data.get("feature_dir", DEFAULT_FEATURES)
    args.splits_file = getattr(args, "splits_file", None) or data.get("splits_file", DEFAULT_SPLITS)
    args.split = getattr(args, "split", None) or data.get("split", "split01")
    args.role = getattr(args, "role", None) or data.get("role", "test")
    args.score_source = getattr(args, "score_source", None) or benchmark.get("score_source", "temporal")
    args.refractory_seconds = (
        getattr(args, "refractory_seconds", None)
        if getattr(args, "refractory_seconds", None) is not None
        else benchmark.get("refractory_seconds", 2.0)
    )
    args.device = getattr(args, "device", None) or ("cuda" if _cuda_available() else "cpu")

    policy = dict(config.get("policy", {}))
    overrides: dict[str, Any] = {}
    if getattr(args, "policy_mode", None):
        overrides["mode"] = args.policy_mode
    if getattr(args, "confidence_threshold", None) is not None:
        overrides["confidence_threshold"] = args.confidence_threshold
    if getattr(args, "persistence_seconds", None) is not None:
        overrides["persistence_seconds"] = args.persistence_seconds
    if getattr(args, "decision_window_seconds", None) is not None:
        overrides["decision_window_seconds"] = args.decision_window_seconds
    if getattr(args, "refractory_seconds", None) is not None:
        overrides["refractory_seconds"] = args.refractory_seconds
    if getattr(args, "frame_skip", None) is not None:
        overrides["frame_skip"] = args.frame_skip
    args.policy_overrides = overrides
    if getattr(args, "policy_file", None):
        payload = json.loads(Path(args.policy_file).read_text(encoding="utf-8"))
        selection = payload.get("selection", payload)
        selected = {
            "mode": selection.get("mode", overrides.get("mode", "fixed")),
            "confidence_threshold": float(selection.get("threshold", 0.7)),
            "persistence_seconds": float(selection.get("persistence_seconds", 0.0)),
        }
        # An explicitly requested override always wins over the selected point.
        for key in ("mode", "confidence_threshold", "persistence_seconds"):
            if key in overrides:
                selected[key] = overrides[key]
        args.policy_overrides = selected
        LOGGER.info("applied the operating point from %s: %s", args.policy_file, selected)
    args.policy_config = policy
    args.front_end = config.get("front_end", {})
    return config


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def summarise(values: list[float] | np.ndarray) -> dict[str, float]:
    """Mean / std / min / max of a list of values (NaN-safe)."""
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def write_result(
    out_dir: str | Path,
    name: str,
    payload: dict[str, Any],
) -> Path:
    """Write a JSON result file, creating the directory if needed."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    LOGGER.info("wrote %s", path)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)
