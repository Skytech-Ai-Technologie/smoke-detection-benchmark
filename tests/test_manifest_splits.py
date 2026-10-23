"""Tests for the manifest and the video-level split construction."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.build_manifest import MANIFEST_COLUMNS, detect_label, detect_session
from src.data.video_split import SplitConfig, assign_splits, build_splits, group_sessions, read_manifest

pytestmark = pytest.mark.dataset


MANIFEST = Path("data/manifests/firesense_videos.csv")
SPLITS = Path("data/manifests/splits.csv")


def _manifest_rows() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        pytest.skip(f"manifest not available: {MANIFEST}")
    return read_manifest(MANIFEST)


def test_manifest_has_the_required_columns():
    rows = _manifest_rows()
    assert rows, "manifest is empty"
    for column in MANIFEST_COLUMNS:
        assert column in rows[0], f"missing column {column}"


def test_video_ids_are_unique():
    rows = _manifest_rows()
    ids = [row["video_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_labels_are_binary():
    rows = _manifest_rows()
    assert {row["label"] for row in rows} <= {"smoke", "negative"}


def test_every_video_has_a_hash_and_a_duration():
    rows = _manifest_rows()
    for row in rows:
        assert len(row["sha256"]) == 64
        assert float(row["duration_sec"]) > 0
        assert int(row["num_frames"]) > 0
        assert float(row["fps"]) > 0


def test_detect_label_and_session():
    root = Path("data/raw/firesense_smoke")
    assert detect_label(root / "pos" / "testpos01.817.avi", root) == "smoke"
    assert detect_label(root / "neg" / "testneg01.807.avi", root) == "negative"
    # Distinct recordings must not be merged into one session.
    assert detect_session(root / "pos" / "testpos01.817.avi", root) != detect_session(
        root / "pos" / "testpos02.818.avi", root
    )
    # ... but two files that differ only by a trailing counter are one session.
    assert detect_session(root / "pos" / "testpos01.817.avi", root) == detect_session(
        root / "pos" / "testpos01.818.avi", root
    )


def test_group_sessions_keeps_sessions_together():
    rows = [
        {"video_id": "a", "session_id": "s1", "label": "smoke"},
        {"video_id": "b", "session_id": "s1", "label": "smoke"},
        {"video_id": "c", "session_id": "s2", "label": "smoke"},
    ]
    groups = group_sessions(rows)
    assert set(groups) == {"s1", "s2"}
    assert len(groups["s1"]) == 2


def test_splits_contain_every_video_exactly_once():
    rows = _manifest_rows()
    if not SPLITS.exists():
        pytest.skip(f"split registry not available: {SPLITS}")
    split_rows = read_manifest(SPLITS)
    split_ids = sorted({row["split_id"] for row in split_rows})
    assert len(split_ids) >= 5, "at least five predefined splits are required"

    for split_id in split_ids:
        videos = [row["video_id"] for row in split_rows if row["split_id"] == split_id]
        assert len(videos) == len(set(videos))
        assert set(videos) == {row["video_id"] for row in rows}


def test_split_roles_are_stratified_by_label():
    rows = _manifest_rows()
    if not SPLITS.exists():
        pytest.skip(f"split registry not available: {SPLITS}")
    split_rows = read_manifest(SPLITS)
    for split_id in sorted({row["split_id"] for row in split_rows}):
        for role in ("train", "validation", "test"):
            subset = [
                row
                for row in split_rows
                if row["split_id"] == split_id and row["role"] == role
            ]
            assert subset, f"{split_id}/{role} is empty"
            labels = {row["label"] for row in subset}
            assert labels <= {"smoke", "negative"}


def test_assign_splits_is_deterministic():
    rows = [
        {"video_id": f"v{i}", "session_id": f"s{i}", "label": "smoke" if i % 2 else "negative"}
        for i in range(20)
    ]
    config = SplitConfig(num_splits=2, seed=7)
    first = assign_splits(rows, config, 0)
    second = assign_splits(rows, config, 0)
    assert {r["video_id"] for r in first["test"]} == {r["video_id"] for r in second["test"]}
    third = assign_splits(rows, config, 1)
    assert {r["video_id"] for r in first["test"]} != {r["video_id"] for r in third["test"]}


def test_build_splits_writes_a_lock_file(tmp_path: Path):
    rows = [
        {"video_id": f"v{i}", "session_id": f"s{i}", "label": "smoke" if i % 2 else "negative"}
        for i in range(20)
    ]
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "session_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    summary = build_splits(manifest, tmp_path, SplitConfig(num_splits=2, seed=1))
    assert len(summary["splits"]) == 2
    assert (tmp_path / "splits_locked.json").exists()
    assert (tmp_path / "lovo.csv").exists()
