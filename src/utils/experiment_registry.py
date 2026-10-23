"""Experiment registry: one immutable, machine-readable record per run.

The registry is the audit trail of the project. Each entry records the git
commit, dataset manifest hash, config hash, seed, device, precision, the exact
command line, and the checkpoint hash, so that a number in the paper can be
traced back to the artefact that produced it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .checkpoints import sha256_file, sha256_json
from .logging import get_logger

__all__ = [
    "ExperimentRecord",
    "git_commit",
    "build_record",
    "experiment_id",
    "results_dir",
    "append_to_registry",
    "read_registry",
    "REGISTRY_FIELDS",
]

LOGGER = get_logger("utils.experiment_registry")

REGISTRY_FIELDS = [
    "experiment_id",
    "timestamp_utc",
    "git_commit",
    "dataset_manifest_sha256",
    "config_sha256",
    "seed",
    "device",
    "precision",
    "command",
    "checkpoint",
    "checkpoint_sha256",
    "split",
    "model",
    "notes",
    "metrics_json",
]


@dataclass
class ExperimentRecord:
    """An immutable description of one experimental run."""

    experiment_id: str
    timestamp_utc: str
    git_commit: str
    dataset_manifest_sha256: str
    config_sha256: str
    seed: int
    device: str
    precision: str
    command: str
    checkpoint: str
    checkpoint_sha256: str
    split: str = ""
    model: str = ""
    notes: str = ""
    metrics_json: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo."""
    cwd = str(Path(repo_root) if repo_root else Path.cwd())
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def device_string() -> str:
    """Describe the compute device used for a run."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def experiment_id(
    model: str,
    *,
    split: str = "",
    seed: int = 0,
    precision: str = "fp32",
    suffix: str = "",
    extra: Iterable[str] = (),
) -> str:
    """Build a unique, human-readable experiment identifier.

    Example: ``tcn_firesense_split01_seed42_fp16``.
    """
    parts = [model]
    if split:
        parts.append(split)
    if seed:
        parts.append(f"seed{seed}")
    if precision and precision != "fp32":
        parts.append(precision)
    parts.extend(str(p) for p in extra if p)
    if suffix:
        parts.append(suffix)
    return "_".join(str(p) for p in parts)


def results_dir(root: str | os.PathLike[str], exp_id: str, *, allow_existing: bool = False) -> Path:
    """Return ``root/<exp_id>`` and refuse to reuse a non-empty directory.

    The specification requires that an experiment directory is never
    overwritten. When ``allow_existing`` is False and the directory already
    contains files, a unique suffix is appended instead of destroying data.
    """
    base = Path(root)
    target = base / exp_id
    if target.exists() and any(target.iterdir()) and not allow_existing:
        target = base / f"{exp_id}__{uuid.uuid4().hex[:8]}"
        LOGGER.warning("experiment directory already populated; using %s", target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_record(
    *,
    exp_id: str,
    config: dict[str, Any] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    manifest_sha256: str | None = None,
    seed: int = 0,
    device: str | None = None,
    precision: str = "fp32",
    command: str | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    split: str = "",
    model: str = "",
    notes: str = "",
    metrics: dict[str, Any] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> ExperimentRecord:
    """Assemble an :class:`ExperimentRecord`."""
    import json

    config_sha = sha256_json(config) if config is not None else "none"
    if manifest_sha256 is None and manifest_path is not None:
        p = Path(manifest_path)
        manifest_sha256 = sha256_file(p) if p.exists() else "missing"

    ckpt_path = str(checkpoint) if checkpoint is not None else ""
    ckpt_sha = ""
    if checkpoint is not None and Path(checkpoint).exists():
        ckpt_sha = sha256_file(checkpoint)

    return ExperimentRecord(
        experiment_id=exp_id,
        timestamp_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(repo_root),
        dataset_manifest_sha256=manifest_sha256 or "none",
        config_sha256=config_sha,
        seed=int(seed),
        device=device or device_string(),
        precision=precision,
        command=command or " ".join(sys.argv),
        checkpoint=ckpt_path,
        checkpoint_sha256=ckpt_sha,
        split=split,
        model=model,
        notes=notes,
        metrics_json=json.dumps(metrics or {}, sort_keys=True),
    )


def append_to_registry(csv_path: str | os.PathLike[str], record: ExperimentRecord) -> Path:
    """Append a record to the CSV registry (creating the header if needed)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.to_dict().get(k, "") for k in REGISTRY_FIELDS})
    LOGGER.info("appended experiment %s to %s", record.experiment_id, path)
    return path


def read_registry(csv_path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Read the registry as a list of dicts (empty list if it does not exist)."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def environment_summary() -> dict[str, Any]:
    """A compact environment fingerprint stored alongside every record."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device_string(),
    }
