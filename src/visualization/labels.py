"""Shared helpers that turn raw artifact names into figure labels.

Every figure label must say *which model* a curve belongs to; artifact file
names (``video_eval_voting_mean_split03_seed42``) never appear in a figure.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["parse_eval_artifact", "pretty_model", "pretty_nonempty", "split_from_name"]

EVAL_PREFIXES = ("frame_eval_", "video_eval_", "calibration_eval_", "streaming_eval_", "ablations_")

MODEL_PRETTY = {
    "tcn": "TCN (proposed)",
    "gru": "GRU",
    "spatial": "Frame-only CNN",
    "voting_mean": "Mean vote (3 s)",
    "voting_median": "Median vote (3 s)",
    "voting_majority": "Majority vote (3 s)",
    "voting_max": "Max vote (3 s)",
    "voting_persistence": "Persistence rule",
    "evidence": "Evidence accumulator",
}

_SEED = re.compile(r"_seed\d+$")
_SPLIT = re.compile(r"(?:^|_)(split\d+)")
_LEGACY = re.compile(r"^(?P<base>[a-z_]+)_(split\d+)$")


def parse_eval_artifact(stem: str) -> dict | None:
    """Parse an evaluation artifact stem into its identity.

    Returns ``None`` for artifacts that are not per-split evaluations (e.g.
    ``sweep_tcn_split01_seed42``), which figures should not treat as models.

    Returns a dict with:

    * ``kind`` -- evaluation kind (``frame``/``video``/``calibration``/...),
    * ``model`` -- raw model tag (``tcn``, ``voting_mean`` ...),
    * ``model_pretty`` -- display name,
    * ``split`` -- ``split01`` ... ``split05``,
    * ``legacy`` -- True when the artifact predates the ``*_tcn_*`` renaming
      (un-tagged splits are the TCN runs of the original pipeline).
    """
    for prefix in EVAL_PREFIXES:
        if stem.startswith(prefix):
            break
    else:
        return None
    kind = prefix[:-1]
    rest = stem[len(prefix):]
    rest = _SEED.sub("", rest)
    match = _SPLIT.search(rest)
    if not match:
        return None
    split = match.group(1)
    legacy = False
    # ``rest`` looks like ``tcn_split01``, ``voting_mean_split03`` or the
    # legacy ``split03`` (no model segment).
    if rest.startswith(f"{split}"):
        legacy = True
        model_tag = "tcn"
    else:
        model_tag = rest[: rest.rfind(f"_{split}")]
    if not model_tag or not re.fullmatch(r"[a-z_]+", model_tag):
        return None
    return {
        "kind": kind,
        "model": model_tag,
        "model_pretty": pretty_model(model_tag),
        "split": split,
        "legacy": legacy,
        "stem": stem,
    }


def pretty_model(tag: str) -> str:
    """Display name of a model tag."""
    return MODEL_PRETTY.get(tag.lower(), tag.replace("_", " "))


def pretty_nonempty(*parts: str, sep: str = " — ") -> str:
    """Join the non-empty parts with ``sep``."""
    return sep.join(part for part in parts if part)


def split_from_name(path: str | Path) -> str | None:
    """Extract the ``split0X`` token of an artifact name."""
    match = _SPLIT.search(str(path))
    return match.group(1) if match else None


def eval_artifact_stem(stem: str) -> dict | None:
    """Backwards-compatible alias; use :func:`parse_eval_artifact`."""
    return parse_eval_artifact(stem)
