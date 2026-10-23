"""Deterministic seeding helpers.

Every experiment records the seed that produced it. This module centralises
seed application so that a run can be reproduced from a single integer.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:  # torch is a hard dependency of the project, but keep the import soft for tooling
    import torch
except Exception:  # pragma: no cover - torch missing only in doc tooling
    torch = None  # type: ignore[assignment]

__all__ = [
    "set_seed",
    "seed_worker",
    "make_generator",
    "seed_everything_report",
    "derive_seed",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and (if available) ``torch``.

    Args:
        seed: Non-negative integer seed.
        deterministic: When True, request deterministic cuDNN kernels. This
            slows the training loop slightly but removes a source of
            run-to-run variance that would otherwise contaminate the
            multi-seed statistics reported in the paper.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Guard against cuBLAS workspace re-use changing reductions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` worker initialiser that keeps workers reproducible."""
    worker_seed = torch.initial_seed() % (2**32) if torch is not None else worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for make_generator")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable child seed from a base seed and arbitrary string parts.

    Used to give each (split, model, seed) combination its own deterministic
    but distinct random stream without any hidden global state.
    """
    import hashlib

    payload = "|".join([str(base_seed), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def seed_everything_report(seed: int) -> dict[str, Any]:
    """Apply ``set_seed`` and return a machine-readable description of the state."""
    set_seed(seed)
    report: dict[str, Any] = {
        "seed": int(seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "numpy_version": np.__version__,
    }
    if torch is not None:
        report["torch_version"] = torch.__version__
        report["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        report["cuda_available"] = bool(torch.cuda.is_available())
    return report
