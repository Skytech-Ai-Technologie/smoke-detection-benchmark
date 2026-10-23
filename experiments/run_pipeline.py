"""Run the complete experiment pipeline for one video split.

The pipeline reproduces one row of every table in the paper:

    Stage B  spatial training (weakly supervised, MIL)      -> spatial model
    Stage C  temporal training (causal TCN)                 -> temporal model
    Stage D  calibration on the validation split             -> calibrated model
    Stage E  operating-point selection on the validation split
    Evaluation  frame level, video level, calibration
    Sweep   the false-alarms-per-hour versus recall curve

Every step writes a machine-readable record into ``results/raw`` and appends
an entry to ``experiments/registry.csv``.

Usage
-----
    python experiments/run_pipeline.py --split split01 --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str], label: str, log_dir: Path) -> int:
    """Run a sub-command and stream its output into a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    print(f"=== {label} ===", flush=True)
    started = time.perf_counter()
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
            if any(key in line for key in ("ERROR", "Traceback", "video metrics", "frame metrics", "selected", "saved checkpoint")):
                print(f"  {line.rstrip()}", flush=True)
        code = process.wait()
    print(f"--- {label} finished in {time.perf_counter() - started:.1f} s (exit {code})", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full pipeline for one split.")
    parser.add_argument("--split", default="split01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--spatial-config", default="configs/baseline_kalman_hsv_cnn.yaml")
    parser.add_argument("--temporal-config", default="configs/temporal_tcn.yaml")
    parser.add_argument("--benchmark-config", default="configs/benchmark_edge.yaml")
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--max-false-alarms-per-hour", type=float, default=1.0)
    parser.add_argument("--calibration", default="isotonic")
    parser.add_argument("--skip-spatial", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if _cuda() else "cpu")
    log_dir = ROOT / "results" / "logs" / f"{args.split}_seed{args.seed}"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_ckpt = out_dir / f"spatial_{args.split}_seed{args.seed}" / "model.pt"
    temporal_ckpt = out_dir / f"tcn_{args.split}_seed{args.seed}" / "model.pt"
    calibrated_ckpt = out_dir / f"tcn_{args.split}_seed{args.seed}" / "model_calibrated.pt"

    steps: list[tuple[str, list[str]]] = []

    if not args.skip_spatial:
        steps.append(
            (
                "train_spatial",
                [
                    PYTHON, "-m", "src.training.train_spatial",
                    "--config", args.spatial_config,
                    "--dataset", "firesense",
                    "--split", args.split,
                    "--seed", str(args.seed),
                    "--device", device,
                    "--out", str(spatial_ckpt),
                ],
            )
        )

    steps.append(
        (
            "train_temporal",
            [
                PYTHON, "-m", "src.training.train_temporal",
                "--config", args.temporal_config,
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split,
                "--seed", str(args.seed),
                "--device", device,
                "--out", str(temporal_ckpt),
            ],
        )
    )
    steps.append(
        (
            "calibrate",
            [
                PYTHON, "-m", "src.training.calibrate",
                "--checkpoint", str(temporal_ckpt),
                "--spatial-checkpoint", str(spatial_ckpt),
                "--method", args.calibration,
                "--split", args.split,
                "--device", device,
                "--out", str(calibrated_ckpt),
            ],
        )
    )
    steps.append(
        (
            "select_thresholds",
            [
                PYTHON, "-m", "src.training.select_thresholds",
                "--model", str(calibrated_ckpt),
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split,
                "--device", device,
                "--max-false-alarms-per-hour", str(args.max_false_alarms_per_hour),
                "--out-dir", args.out_dir,
                "--tag", f"thresholds_tcn_{args.split}_seed{args.seed}",
            ],
        )
    )

    policy_file = out_dir / f"thresholds_tcn_{args.split}_seed{args.seed}.json"
    for label, command, extra in [
        (
            "evaluate_frame",
            [
                PYTHON, "-m", "src.evaluation.evaluate_frame",
                "--config", args.benchmark_config,
                "--model", str(calibrated_ckpt),
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split, "--role", "test",
                "--policy-file", str(policy_file),
                "--tag", f"frame_eval_tcn_{args.split}_seed{args.seed}",
                "--out-dir", args.out_dir,
            ],
            [],
        ),
        (
            "evaluate_video",
            [
                PYTHON, "-m", "src.evaluation.evaluate_video",
                "--config", args.benchmark_config,
                "--model", str(calibrated_ckpt),
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split, "--role", "test",
                "--policy-file", str(policy_file),
                "--tag", f"video_eval_tcn_{args.split}_seed{args.seed}",
                "--out-dir", args.out_dir,
            ],
            [],
        ),
        (
            "evaluate_calibration",
            [
                PYTHON, "-m", "src.evaluation.evaluate_calibration",
                "--config", args.benchmark_config,
                "--model", str(calibrated_ckpt),
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split, "--role", "test",
                "--tag", f"calibration_eval_tcn_{args.split}_seed{args.seed}",
                "--out-dir", args.out_dir,
            ],
            [],
        ),
        (
            "sweep_curve",
            [
                PYTHON, "-m", "src.evaluation.sweep_curve",
                "--models", f"spatial=none", f"tcn={calibrated_ckpt}",
                "--spatial-checkpoint", str(spatial_ckpt),
                "--split", args.split, "--role", "test",
                "--device", device,
                "--out-dir", args.out_dir,
                "--tag", f"sweep_tcn_{args.split}_seed{args.seed}",
            ],
            [],
        ),
    ]:
        steps.append((label, command + extra))

    summary: dict[str, object] = {"split": args.split, "seed": args.seed, "steps": {}}
    for label, command in steps:
        code = run(command, label, log_dir)
        summary["steps"][label] = {"exit_code": code, "command": " ".join(command)}
        if code != 0:
            print(f"!! {label} failed with exit code {code}; stopping", flush=True)
            (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return code

    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("pipeline finished", flush=True)
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


if __name__ == "__main__":
    sys.exit(main())
