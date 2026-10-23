"""Re-run the per-split baseline operating-point sweeps.

Equivalent to the (fixed) ``run_baselines.py`` sweep step, but runnable
standalone so that a partial run can be resumed:

    python tools/run_sweep_baselines.py                # all splits
    python tools/run_sweep_baselines.py split01 split04  # subset
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS = [
    "spatial",
    "voting_mean",
    "voting_median",
    "voting_majority",
    "voting_max",
    "voting_persistence",
]


def main() -> int:
    splits = sys.argv[1:] or [f"split0{i}" for i in range(1, 6)]
    device = "cuda"
    exit_codes: dict[str, int] = {}
    for split in splits:
        spatial = ROOT / "results" / "raw" / f"spatial_{split}_seed42" / "model.pt"
        command = [
            PYTHON, "-m", "src.evaluation.sweep_curve",
            "--models", *(f"{name}=none@{name}" for name in VARIANTS),
            "--spatial-checkpoint", str(spatial),
            "--split", split,
            "--role", "test",
            "--device", device,
            "--out-dir", "results/raw",
            "--tag", f"sweep_baselines_{split}_seed42",
        ]
        started = time.perf_counter()
        print(f"=== sweep_baselines {split} ===", flush=True)
        result = subprocess.run(command, cwd=str(ROOT))
        print(
            f"--- sweep_baselines {split} finished in {time.perf_counter() - started:.1f} s "
            f"(exit {result.returncode})",
            flush=True,
        )
        exit_codes[split] = result.returncode
    bad = {s: c for s, c in exit_codes.items() if c != 0}
    if bad:
        print(f"failed sweeps: {bad}", flush=True)
        return 1
    print("all sweeps finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
