"""Checkpoint IO with content hashing.

Hashes matter for reproducibility: a paper claim must be traceable to the
exact weights that produced the reported numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from .logging import get_logger

__all__ = [
    "sha256_file",
    "sha256_bytes",
    "sha256_json",
    "save_checkpoint",
    "load_checkpoint",
    "Checkpoint",
]

LOGGER = get_logger("utils.checkpoints")
_CHUNK = 1024 * 1024


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of a canonical JSON serialisation of ``obj``."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))


class Checkpoint:
    """A loaded checkpoint plus its provenance metadata."""

    def __init__(self, path: Path, payload: dict[str, Any], sha256: str) -> None:
        self.path = path
        self.payload = payload
        self.sha256 = sha256

    @property
    def model(self) -> torch.nn.Module | None:
        return self.payload.get("model")

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.payload.get("meta", {}))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Checkpoint(path={self.path}, sha256={self.sha256[:12]}...)"


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    *,
    meta: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> str:
    """Save a checkpoint and return its SHA-256.

    Args:
        overwrite: When False, an existing file at ``path`` raises
            ``FileExistsError``. The experiment registry uses unique ids so
            that historical runs are never overwritten.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict() if hasattr(model, "state_dict") else dict(model),
        "meta": dict(meta or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    shutil.move(str(tmp), str(target))
    digest = sha256_file(target)
    LOGGER.info("saved checkpoint %s (sha256=%s)", target, digest[:16])
    return digest


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> Checkpoint:
    """Load a checkpoint, verifying the file can be read."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)!r}")
    digest = sha256_file(source)
    LOGGER.info("loaded checkpoint %s (sha256=%s)", source, digest[:16])
    return Checkpoint(source, payload, digest)
