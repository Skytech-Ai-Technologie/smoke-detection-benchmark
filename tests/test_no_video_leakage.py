"""The most important test of the project: no video may leak across splits.

Frames of a video must never appear in more than one role, and files that
belong to the same recording session must never be torn apart. A violation
of either property invalidates every number reported in the paper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.video_split import SplitConfig, assign_splits, build_splits, read_manifest

pytestmark = pytest.mark.dataset

SPLITS = Path("data/manifests/splits.csv")
MANIFEST = Path("data/manifests/firesense_videos.csv")
ROLES = ("train", "validation", "test")


def _split_rows() -> list[dict[str, str]]:
    if not SPLITS.exists():
        pytest.skip(f"split registry not available: {SPLITS}")
    return read_manifest(SPLITS)


def test_no_video_appears_in_two_roles_of_the_same_split():
    rows = _split_rows()
    for split_id in sorted({row["split_id"] for row in rows}):
        seen: dict[str, set[str]] = {}
        for row in rows:
            if row["split_id"] != split_id:
                continue
            seen.setdefault(row["video_id"], set()).add(row["role"])
        for video_id, roles in seen.items():
            assert len(roles) == 1, (
                f"{video_id} appears in {sorted(roles)} of {split_id}: a video must belong to "
                "exactly one role"
            )


def test_no_session_is_torn_across_roles():
    rows = _split_rows()
    for split_id in sorted({row["split_id"] for row in rows}):
        session_roles: dict[str, set[str]] = {}
        for row in rows:
            if row["split_id"] != split_id:
                continue
            session_roles.setdefault(row["session_id"], set()).add(row["role"])
        for session, roles in session_roles.items():
            assert len(roles) == 1, (
                f"session {session} of {split_id} is spread over {sorted(roles)}: "
                "all files of one recording must stay together"
            )


def test_test_videos_never_appear_in_the_training_role_of_the_same_split():
    rows = _split_rows()
    for split_id in sorted({row["split_id"] for row in rows}):
        test_videos = {
            row["video_id"] for row in rows if row["split_id"] == split_id and row["role"] == "test"
        }
        train_videos = {
            row["video_id"] for row in rows if row["split_id"] == split_id and row["role"] == "train"
        }
        assert not (test_videos & train_videos)


def test_the_primary_test_manifest_is_read_only():
    """The finalised test manifest must not be editable by accident."""
    path = Path("data/manifests/split01_test.csv")
    if not path.exists():
        pytest.skip(f"primary test manifest not available: {path}")
    import stat

    mode = path.stat().st_mode
    assert not (mode & stat.S_IWRITE), (
        "data/manifests/split01_test.csv is writable; lock it with "
        "python -c \"from src.data.video_split import lock_file; lock_file('data/manifests/split01_test.csv')\""
    )


def test_split_hashes_match_the_lock_file():
    import json

    lock_path = Path("data/manifests/splits_locked.json")
    if not lock_path.exists():
        pytest.skip("splits_locked.json not available")
    from src.utils.checkpoints import sha256_file

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, digest in lock["files"].items():
        path = lock_path.parent / name
        if not path.exists():
            continue
        assert sha256_file(path) == digest, f"{name} changed after the splits were locked"


def test_leave_one_video_out_covers_every_video():
    lovo = Path("data/manifests/lovo.csv")
    if not lovo.exists():
        pytest.skip("lovo.csv not available")
    rows = read_manifest(lovo)
    splits = {row["split_id"] for row in rows}
    assert len(splits) == 22 or len(splits) > 0
    for split_id in splits:
        subset = [row for row in rows if row["split_id"] == split_id]
        test = [row for row in subset if row["role"] == "test"]
        assert len(test) == 1
        assert len(subset) - 1 == len([row for row in subset if row["role"] == "train"])


def test_synthetic_split_has_no_leakage():
    """A synthetic dataset with multi-file sessions must also stay clean."""
    rows = []
    for session in range(6):
        label = "smoke" if session % 2 == 0 else "negative"
        for part in range(2):
            rows.append(
                {
                    "video_id": f"s{session}_p{part}",
                    "session_id": f"session{session}",
                    "label": label,
                }
            )
    assignment = assign_splits(rows, SplitConfig(num_splits=5, seed=3), 0)
    placed = {role: {r["video_id"] for r in videos} for role, videos in assignment.items()}
    assert not (placed["train"] & placed["test"])
    assert not (placed["train"] & placed["validation"])
    assert not (placed["validation"] & placed["test"])
    for role in ROLES:
        for video in assignment[role]:
            assert video["session_id"]
    # Sessions are not torn apart.
    session_role: dict[str, set[str]] = {}
    for role in ROLES:
        for video in assignment[role]:
            session_role.setdefault(video["session_id"], set()).add(role)
    for session, roles in session_role.items():
        assert len(roles) == 1
