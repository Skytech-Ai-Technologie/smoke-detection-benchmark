"""Video-level splitting.

**All primary splits are made at the video level.** Frames of a video never
appear in more than one role, and files that belong to the same recording
session are kept together. This module is the single place where the split
logic lives, so the leakage test in ``tests/test_no_video_leakage.py`` can
verify exactly this property.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils.checkpoints import sha256_file
from ..utils.logging import get_logger, setup_logging

__all__ = [
    "SplitConfig",
    "read_manifest",
    "write_manifest",
    "group_sessions",
    "assign_splits",
    "build_splits",
    "lock_file",
    "unlock_file",
    "main",
]

LOGGER = get_logger("data.video_split")

ROLES = ("train", "validation", "test")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> Path:
    """Write rows to a manifest CSV, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return out


def group_sessions(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group manifest rows by ``session_id`` (falling back to ``video_id``)."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("session_id") or "").strip() or row["video_id"]
        groups[key].append(row)
    return dict(groups)


class SplitConfig:
    """Parameters of the split generator."""

    def __init__(
        self,
        num_splits: int = 5,
        test_fraction: float = 0.30,
        validation_fraction: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.num_splits = int(num_splits)
        self.test_fraction = float(test_fraction)
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)


def _class_of(row: dict[str, str]) -> str:
    label = (row.get("label") or "unknown").strip().lower()
    return "smoke" if label in ("smoke", "positive", "1", "true") else "negative"


def _take_sessions(
    sessions: list[tuple[str, list[dict[str, str]]]],
    target: int,
    rng: np.random.Generator,
) -> tuple[list[str], int]:
    """Greedily select sessions until ``target`` videos are collected."""
    order = rng.permutation(len(sessions))
    chosen: list[str] = []
    count = 0
    for idx in order:
        key, videos = sessions[idx]
        chosen.append(key)
        count += len(videos)
        if count >= target:
            break
    return chosen, count


def assign_splits(
    rows: list[dict[str, str]],
    config: SplitConfig,
    split_index: int,
) -> dict[str, list[dict[str, str]]]:
    """Assign every video of one split to a role.

    The assignment is stratified by label and grouped by session, so both
    classes are represented in each role in (approximately) equal proportion
    and no recording session is ever torn apart.
    """
    groups = group_sessions(rows)
    by_class: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for key, videos in groups.items():
        by_class[_class_of(videos[0])].append((key, videos))

    rng = np.random.default_rng(config.seed * 1000 + split_index)
    assignment: dict[str, list[dict[str, str]]] = {role: [] for role in ROLES}

    for label, sessions in by_class.items():
        order = rng.permutation(len(sessions))
        ordered = [sessions[i] for i in order]
        total_videos = sum(len(v) for _, v in ordered)

        # 1) test set
        test_target = max(1, int(round(config.test_fraction * total_videos)))
        test_sessions: list[tuple[str, list[dict[str, str]]]] = []
        used = 0
        for key, videos in ordered:
            if used >= test_target:
                break
            test_sessions.append((key, videos))
            used += len(videos)

        # 2) validation set from the remaining sessions
        remaining = [(k, v) for k, v in ordered if k not in {s[0] for s in test_sessions}]
        remaining_videos = sum(len(v) for _, v in remaining)
        val_target = max(1, int(round(config.validation_fraction * remaining_videos))) if remaining else 0
        val_sessions: list[tuple[str, list[dict[str, str]]]] = []
        used = 0
        for key, videos in remaining:
            if used >= val_target:
                break
            val_sessions.append((key, videos))
            used += len(videos)

        test_keys = {s[0] for s in test_sessions}
        val_keys = {s[0] for s in val_sessions}

        for key, videos in ordered:
            if key in test_keys:
                role = "test"
            elif key in val_keys:
                role = "validation"
            else:
                role = "train"
            assignment[role].extend(videos)

    # Safety net: any video that was not covered ends up in train.
    assigned = {v["video_id"] for role in ROLES for v in assignment[role]}
    for video in rows:
        if video["video_id"] not in assigned:
            assignment["train"].append(video)

    for role in ROLES:
        assignment[role].sort(key=lambda r: r["video_id"])
    return assignment


def build_splits(
    manifest_path: str | Path,
    out_dir: str | Path,
    config: SplitConfig | None = None,
    *,
    primary_split: str = "split01",
    write_lovo: bool = True,
) -> dict[str, Any]:
    """Create the predefined splits and write them to ``out_dir``.

    Returns:
        A summary dictionary describing the generated splits.
    """
    cfg = config or SplitConfig()
    manifest = Path(manifest_path)
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())

    # A previously locked split must be unlocked before it can be rewritten.
    for existing in sorted(out.glob("*.csv")):
        unlock_file(existing)

    all_assignments: dict[str, dict[str, list[dict[str, str]]]] = {}
    for i in range(cfg.num_splits):
        split_id = f"split{i + 1:02d}"
        assignment = assign_splits(rows, cfg, i)
        all_assignments[split_id] = assignment

        for role in ROLES:
            rows_out = [dict(v, split=split_id) for v in assignment[role]]
            write_manifest(out / f"{split_id}_{role}.csv", rows_out, columns)

    # Flat registry of all splits.
    registry_rows: list[dict[str, Any]] = []
    for split_id, assignment in all_assignments.items():
        for role in ROLES:
            for video in assignment[role]:
                registry_rows.append(
                    {
                        "split_id": split_id,
                        "role": role,
                        "video_id": video["video_id"],
                        "dataset": video.get("dataset", ""),
                        "label": video.get("label", ""),
                        "session_id": video.get("session_id", ""),
                        "sha256": video.get("sha256", ""),
                    }
                )
    write_manifest(
        out / "splits.csv",
        registry_rows,
        ["split_id", "role", "video_id", "dataset", "label", "session_id", "sha256"],
    )

    # Convenience copies for the primary split.
    primary = all_assignments.get(primary_split)
    if primary is not None:
        for role in ROLES:
            write_manifest(out / f"{role}.csv", [dict(v, split=primary_split) for v in primary[role]], columns)

    # Leave-one-video-out sensitivity analysis.
    if write_lovo:
        lovo_rows: list[dict[str, Any]] = []
        for video in rows:
            for other in rows:
                if other["video_id"] == video["video_id"]:
                    continue
                lovo_rows.append(
                    {
                        "split_id": f"lovo_{video['video_id']}",
                        "role": "train",
                        "video_id": other["video_id"],
                        "dataset": other.get("dataset", ""),
                        "label": other.get("label", ""),
                        "session_id": other.get("session_id", ""),
                        "sha256": other.get("sha256", ""),
                    }
                )
            lovo_rows.append(
                {
                    "split_id": f"lovo_{video['video_id']}",
                    "role": "test",
                    "video_id": video["video_id"],
                    "dataset": video.get("dataset", ""),
                    "label": video.get("label", ""),
                    "session_id": video.get("session_id", ""),
                    "sha256": video.get("sha256", ""),
                }
            )
        write_manifest(
            out / "lovo.csv",
            lovo_rows,
            ["split_id", "role", "video_id", "dataset", "label", "session_id", "sha256"],
        )

    # Lock the primary test manifest: it must not change after finalisation.
    lock_info: dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "num_splits": cfg.num_splits,
        "test_fraction": cfg.test_fraction,
        "validation_fraction": cfg.validation_fraction,
        "primary_split": primary_split,
        "files": {},
    }
    for path in sorted(out.glob("*.csv")):
        lock_info["files"][path.name] = sha256_file(path)
    (out / "splits_locked.json").write_text(json.dumps(lock_info, indent=2), encoding="utf-8")

    # The test manifests are read-only from now on.
    for path in sorted(out.glob("*_test.csv")):
        lock_file(path)

    summary: dict[str, Any] = {"splits": {}, "lock": lock_info}
    for split_id, assignment in all_assignments.items():
        summary["splits"][split_id] = {
            role: {
                "n": len(assignment[role]),
                "n_positive": sum(1 for v in assignment[role] if _class_of(v) == "smoke"),
                "n_negative": sum(1 for v in assignment[role] if _class_of(v) == "negative"),
            }
            for role in ROLES
        }
    return summary


def lock_file(path: str | Path) -> None:
    """Mark a file read-only (protects the finalised test manifest)."""
    target = Path(path)
    if target.exists():
        target.chmod(stat.S_IREAD)
        LOGGER.info("locked %s (read-only)", target)


def unlock_file(path: str | Path) -> None:
    """Remove the read-only flag (only to regenerate a split)."""
    target = Path(path)
    if target.exists():
        target.chmod(stat.S_IWRITE)
        LOGGER.info("unlocked %s", target)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create video-level splits.")
    parser.add_argument("--manifest", required=True, help="video manifest CSV")
    parser.add_argument("--out-dir", default="data/manifests", help="output directory")
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-split", default="split01")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    config = SplitConfig(
        num_splits=args.num_splits,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    summary = build_splits(
        args.manifest,
        args.out_dir,
        config,
        primary_split=args.primary_split,
    )
    for split_id, roles in summary["splits"].items():
        LOGGER.info(
            "%s: train=%d(+%d pos) val=%d(+%d pos) test=%d(+%d pos)",
            split_id,
            roles["train"]["n"], roles["train"]["n_positive"],
            roles["validation"]["n"], roles["validation"]["n_positive"],
            roles["test"]["n"], roles["test"]["n_positive"],
        )
    LOGGER.info("split hashes recorded in %s", Path(args.out_dir) / "splits_locked.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
