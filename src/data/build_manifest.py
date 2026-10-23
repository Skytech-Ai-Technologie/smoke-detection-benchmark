"""Build the dataset manifests.

A manifest is a CSV with one row per *video* (not per frame). It records the
provenance of every item so that a manifest hash can be used as a dataset
version identifier in the experiment registry.

Expected columns::

    video_id,dataset,path,label,num_frames,fps,width,height,duration_sec,
    sha256,split,session_id,onset_frame,smoke_visibility,scene_type,source_url
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from ..utils.checkpoints import sha256_file
from ..utils.logging import get_logger, setup_logging
from .inspect import iter_videos, video_metadata

__all__ = [
    "MANIFEST_COLUMNS",
    "detect_label",
    "detect_session",
    "load_annotations",
    "build_manifest",
    "main",
]

LOGGER = get_logger("data.build_manifest")

MANIFEST_COLUMNS = [
    "video_id",
    "dataset",
    "path",
    "label",
    "num_frames",
    "fps",
    "width",
    "height",
    "duration_sec",
    "sha256",
    "split",
    "session_id",
    "onset_frame",
    "smoke_visibility",
    "scene_type",
    "source_url",
    "video_name",
]

# Keyword table used to infer the label from the file path.
# Order matters: the first match wins, so "no_smoke" is checked before "smoke".
LABEL_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("no_smoke", "nonsmoke", "no-smoke", "without_smoke", "negative", "neg", "background"), "negative"),
    (("smoke", "fire", "positive", "pos"), "smoke"),
]

SESSION_KEYWORDS = ("session", "seq", "take", "clip", "part", "run")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def detect_label(path: Path, root: Path) -> str:
    """Infer a binary smoke label from the file path.

    The heuristics are conservative: the label is taken from the *directory*
    structure first and only then from the file name, because directories in
    the public archives are curated while file names are not.
    """
    rel_parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    for part in rel_parts:
        for keywords, label in LABEL_KEYWORDS:
            if any(kw in part for kw in keywords):
                return label
    stem = _normalise(path.stem)
    for keywords, label in LABEL_KEYWORDS:
        for kw in keywords:
            if kw in stem:
                return label
    return "unknown"


def detect_session(path: Path, root: Path) -> str:
    """Infer a recording-session identifier.

    Files that share a session must stay in the same split, otherwise frames
    from a single recording can leak across the train/test boundary. The
    heuristic is deliberately conservative: it only groups files when the
    name explicitly marks a session, or when the file names differ solely by
    a trailing counter (``clip01.807`` / ``clip01.808`` are the same
    recording, ``clip01`` and ``clip02`` are not).
    """
    rel = path.relative_to(root)
    stem = _normalise(path.stem)

    # 1) explicit session markers inside the file name.
    for keyword in SESSION_KEYWORDS:
        index = stem.find(keyword)
        if index > 0:
            return stem[:index]

    # 2) directory names that mark a session.
    for part in rel.parts[:-1]:
        norm = _normalise(part)
        for keyword in SESSION_KEYWORDS:
            if keyword in norm:
                return norm

    # 3) strip a trailing counter: "clip01_807" -> "clip01".
    match = re.match(r"^(.*?)[._]?(\d+)$", stem)
    if match and match.group(1):
        return match.group(1)
    return stem


def load_annotations(root: str | Path) -> dict[str, dict[str, Any]]:
    """Find and parse annotation files shipped with a dataset.

    Supports CSV files with a video column and an onset / start column, and
    JSON files mapping a video name to an annotation record.

    Returns:
        A mapping ``video_stem -> {"onset_frame": int | None, ...}``.
    """
    base = Path(root)
    out: dict[str, dict[str, Any]] = {}

    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".txt"}:
            continue
        if path.suffix.lower() == ".csv":
            try:
                with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue
                    fields = {f.strip().lower(): f for f in reader.fieldnames}
                    video_col = next(
                        (fields[k] for k in ("video", "video_id", "filename", "file", "name") if k in fields),
                        None,
                    )
                    onset_col = next(
                        (
                            fields[k]
                            for k in ("onset_frame", "onset", "start_frame", "start", "smoke_start")
                            if k in fields
                        ),
                        None,
                    )
                    if video_col is None or onset_col is None:
                        continue
                    vis_col = next(
                        (fields[k] for k in ("visibility", "smoke_visibility", "level") if k in fields), None
                    )
                    for row in reader:
                        stem = Path(str(row[video_col])).stem
                        onset_raw = str(row.get(onset_col, "")).strip()
                        onset = int(float(onset_raw)) if onset_raw else None
                        record: dict[str, Any] = {"onset_frame": onset}
                        if vis_col:
                            record["smoke_visibility"] = str(row.get(vis_col, "")).strip()
                        out[stem] = record
            except Exception as exc:  # pragma: no cover - malformed annotation files
                LOGGER.warning("could not parse annotations from %s: %s", path, exc)
        elif path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, dict):
                        stem = Path(str(key)).stem
                        onset = value.get("onset_frame", value.get("onset"))
                        try:
                            onset = int(onset) if onset is not None else None
                        except (TypeError, ValueError):
                            onset = None
                        out[stem] = {
                            "onset_frame": onset,
                            "smoke_visibility": value.get("visibility", value.get("smoke_visibility", "")),
                        }
    return out


def _build_video_id(dataset: str, path: Path, root: Path, index: int) -> str:
    stem = _normalise(path.stem)
    return f"{_normalise(dataset)}_{stem}" if stem else f"{_normalise(dataset)}_{index:04d}"


def build_manifest(
    root: str | Path,
    dataset: str,
    *,
    out_path: str | Path | None = None,
    source_url: str = "",
    hash_files: bool = True,
) -> list[dict[str, Any]]:
    """Build the video manifest for a dataset.

    Args:
        root: Directory that contains the extracted dataset.
        dataset: Dataset identifier used as a name prefix.
        out_path: Optional CSV output path.
        source_url: Provenance URL recorded in every row.
        hash_files: Compute the SHA-256 of every video (slow for large sets).

    Returns:
        The list of manifest rows.
    """
    base = Path(root)
    videos = iter_videos(base)
    if not videos:
        LOGGER.warning("no videos found under %s", base)
        return []

    annotations = load_annotations(base)
    rows: list[dict[str, Any]] = []

    for index, path in enumerate(videos):
        meta = video_metadata(path)
        label = detect_label(path, base)
        session = detect_session(path, base)
        video_id = _build_video_id(dataset, path, base, index)
        annotation = annotations.get(path.stem, {})

        digest = ""
        if hash_files:
            digest = sha256_file(path)

        rows.append(
            {
                "video_id": video_id,
                "dataset": dataset,
                "path": str(path).replace("\\", "/"),
                "label": label,
                "num_frames": meta.get("num_frames", 0),
                "fps": meta.get("fps", 0.0),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "duration_sec": meta.get("duration_sec", 0.0),
                "sha256": digest,
                "split": "",
                "session_id": session,
                "onset_frame": annotation.get("onset_frame", ""),
                "smoke_visibility": annotation.get("smoke_visibility", ""),
                "scene_type": "",
                "source_url": source_url,
                "video_name": path.name,
            }
        )

    # Guarantee unique ids even if two files share a stem.
    seen: dict[str, int] = {}
    for row in rows:
        vid = row["video_id"]
        if vid in seen:
            seen[vid] += 1
            row["video_id"] = f"{vid}_{seen[vid]}"
        else:
            seen[vid] = 0

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
        LOGGER.info("wrote %d rows to %s", len(rows), out)

    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build a dataset manifest.")
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. firesense")
    parser.add_argument("--root", default="data/raw", help="directory with the extracted data")
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument("--source-url", default="", help="provenance URL recorded in the manifest")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 computation")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    out = args.out or f"data/manifests/{args.dataset}_videos.csv"
    rows = build_manifest(
        args.root,
        args.dataset,
        out_path=out,
        source_url=args.source_url,
        hash_files=not args.no_hash,
    )
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
    LOGGER.info("manifest summary: %s", labels)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
