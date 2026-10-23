"""Inspect and clean up duplicate quantization report folders.

Prints a summary of each ``results/raw/quantized_*`` directory and, with
``--delete-stale``, removes folders whose report duplicates a newer/complete
run (keeps the canonical ``quantized_int8_seed42`` / ``quantized_fp16_seed42``
directories).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def main() -> int:
    delete = "--delete-stale" in sys.argv
    dirs = sorted(RAW.glob("quantized_*"))
    seen_sha: dict[str, Path] = {}
    for d in dirs:
        report = d / "quantization_report.json"
        if not report.exists():
            print(f"{d.name}: NO REPORT")
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        sha = payload.get("checkpoint_sha256", "")
        sizes = payload.get("sizes", {})
        note = payload.get("comparison_note", "")
        keep = d.name in ("quantized_int8_seed42", "quantized_fp16_seed42")
        duplicate_of = seen_sha.get(sha)
        stale = (not keep) and duplicate_of is not None
        print(
            f"{d.name}: sha={sha[:8]} sizes={sizes} note={note!r} "
            f"{'canonical' if keep else ('duplicate of ' + duplicate_of.name if duplicate_of else 'unique')}"
        )
        if delete and stale:
            print(f"  -> deleting {d.name}")
            shutil.rmtree(d)
        seen_sha.setdefault(sha, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
