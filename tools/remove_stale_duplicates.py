"""Remove stale duplicate experiment directories.

Duplicate run folders (``<name>__12ab34cd``) accumulate when the same
experiment id is regenerated. When their ``calibration_report.json`` /
``quantization_report.json`` duplicates a canonical folder's record they are
stale leftovers of failed attempts and are deleted. Run with ``--dry`` to
print what would be removed.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

HASH_SUFFIX = re.compile(r"^(?P<base>.+)__(?P<hash>[0-9a-f]{8})$")
REPORT_FILES = ("calibration_report.json", "quantization_report.json")


def signature(directory: Path) -> str | None:
    """A fingerprint of what a report folder actually contains."""
    for name in REPORT_FILES:
        path = directory / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        keys = payload.get("checkpoint_sha256") or payload.get("source_sha256") or ""
        method = payload.get("selected_method") or payload.get("mode") or ""
        return f"{name}|{keys}|{method}"
    return None


def main() -> int:
    dry = "--dry" in sys.argv
    canonical_sigs: dict[str, Path] = {}
    everything = {d: signature(d) for d in sorted(RAW.glob("*__*") if (RAW / "__").exists() else [])
                  + sorted([p for p in RAW.iterdir() if p.is_dir() and HASH_SUFFIX.match(p.name)])}
    for full_dir in sorted(RAW.iterdir()):
        if not full_dir.is_dir():
            continue
        match = HASH_SUFFIX.match(full_dir.name)
        if not match:
            continue
        sig = signature(full_dir)
        if sig is None:
            print(f"KEEP (no report): {full_dir.name}")
            continue
        base = match.group("base")
        reference = RAW / base
        reference_sig = signature(reference) if reference.exists() else None
        if sig == reference_sig:
            print(f"STALE duplicate: {full_dir.name} (same content as {base})")
            if not dry:
                shutil.rmtree(full_dir)
        else:
            print(f"KEEP (differs): {full_dir.name} {sig} != {reference_sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
