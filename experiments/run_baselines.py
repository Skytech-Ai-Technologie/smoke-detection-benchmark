"""Run the *reference* models of the benchmark for one split.

The reference models are the ones a temporal model has to beat:

* **spatial** -- the frame-only CNN: the calibrated frame probability of the
  strongest candidate, with a fixed threshold (this is the "CNN only"
  baseline of the ablation list),
* **voting_mean / median / majority / max / persistence** -- the fixed
  temporal rules of Model 2 of the specification.

For every model the operating point is selected on the **validation** split
and then applied unchanged to the **test** split; the operating-point sweep
produces the false-alarms-per-hour versus recall curve of each model, which
is the basis of the matched-recall comparison.

Usage
-----
    python experiments/run_baselines.py --split split01 --seed 42 \\
        --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS = {
    "spatial": {"temporal_kind": "none", "voting": {}},
    "voting_mean": {"temporal_kind": "voting", "voting": {"mode": "mean"}},
    "voting_median": {"temporal_kind": "voting", "voting": {"mode": "median"}},
    "voting_majority": {"temporal_kind": "voting", "voting": {"mode": "majority"}},
    "voting_max": {"temporal_kind": "voting", "voting": {"mode": "max"}},
    "voting_persistence": {"temporal_kind": "voting", "voting": {"mode": "persistence"}},
}


def run(command: list[str], label: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    print(f"=== {label} ===", flush=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if any(key in line for key in ("ERROR", "Traceback", "video metrics", "selected operating")):
                print(f"  {line.rstrip()}", flush=True)
        code = process.wait()
    print(f"--- {label} exit {code}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reference models for one split.")
    parser.add_argument("--split", default="split01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--benchmark-config", default="configs/benchmark_edge.yaml")
    parser.add_argument("--max-false-alarms-per-hour", type=float, default=1.0)
    args = parser.parse_args()

    device = args.device or ("cuda" if _cuda() else "cpu")
    log_dir = ROOT / "results" / "logs" / f"{args.split}_seed{args.seed}_baselines"
    out_dir = ROOT / args.out_dir

    for name, spec in VARIANTS.items():
        # The voting rule is selected on the validation split with the
        # detector's own configuration; the "none" temporal kind is the
        # frame-only CNN.
        policy_file = out_dir / f"thresholds_{name}_{args.split}_seed{args.seed}.json"
        video_file = out_dir / f"video_eval_{name}_{args.split}_seed{args.seed}.json"

        # 1) operating point on the validation split
        code = run(
            [
                PYTHON, "-m", "src.training.select_thresholds",
                "--model", "none",
                "--spatial-checkpoint", args.spatial_checkpoint,
                "--split", args.split, "--device", device,
                "--max-false-alarms-per-hour", str(args.max_false_alarms_per_hour),
                "--out-dir", args.out_dir,
                "--tag", f"thresholds_{name}_{args.split}_seed{args.seed}",
                "--variant", name,
            ],
            f"select_thresholds_{name}",
            log_dir,
        )
        if code != 0:
            return code

        # 2) evaluation on the test split
        code = run(
            [
                PYTHON, "-m", "src.evaluation.evaluate_video",
                "--config", args.benchmark_config,
                "--model", "none",
                "--spatial-checkpoint", args.spatial_checkpoint,
                "--split", args.split, "--role", "test",
                "--policy-file", str(policy_file),
                "--tag", f"video_eval_{name}_{args.split}_seed{args.seed}",
                "--out-dir", args.out_dir,
                "--variant", name,
            ],
            f"evaluate_video_{name}",
            log_dir,
        )
        if code != 0:
            return code

    # 3) the operating-point sweeps of all the reference models
    models = [f"{name}=none@{name}" for name in VARIANTS]
    code = run(
        [
            PYTHON, "-m", "src.evaluation.sweep_curve",
            "--models", *models,            "--spatial-checkpoint", args.spatial_checkpoint,
            "--split", args.split, "--role", "test", "--device", device,
            "--out-dir", args.out_dir,
            "--tag", f"sweep_baselines_{args.split}_seed{args.seed}",
        ],
        "sweep_baselines",
        log_dir,
    )
    return code


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__":
    sys.exit(main())
