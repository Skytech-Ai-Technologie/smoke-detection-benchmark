"""Produce every table, figure and statistical test of the benchmark.

The script assumes that the five split pipelines have already been run
(``experiments/run_pipeline.py``) and that the reference models have been run
for the split that carries the detailed ablation
(``experiments/run_baselines.py``).

It produces

* ``results/tables/summary_over_splits.csv`` -- the mean and the standard
  deviation over the five splits,
* ``results/tables/table1_main_accuracy.csv`` ... ``table5_ablations.csv``,
* ``results/figures/*.png`` -- the paper figures,
* ``results/tables/statistics_*.csv`` -- the paired statistical tests.

Usage
-----
    python experiments/run_reporting.py --split split01 --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str], label: str) -> int:
    print(f"=== {label} ===", flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    print(f"--- {label} exit {result.returncode}", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce the tables, figures and statistics.")
    parser.add_argument("--split", default="split01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--temporal-checkpoint", default=None)
    parser.add_argument("--calibrated-checkpoint", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    raw = ROOT / "results" / "raw"
    spatial = args.spatial_checkpoint or str(
        raw / f"spatial_{args.split}_seed{args.seed}" / "model.pt"
    )
    temporal = args.temporal_checkpoint or str(raw / f"tcn_{args.split}_seed{args.seed}" / "model.pt")
    calibrated = args.calibrated_checkpoint or str(
        raw / f"tcn_{args.split}_seed{args.seed}" / "model_calibrated.pt"
    )

    if not args.skip_baselines and Path(spatial).exists():
        run(
            [
                PYTHON, "experiments/run_baselines.py",
                "--split", args.split, "--seed", str(args.seed),
                "--spatial-checkpoint", spatial,
            ],
            "baselines",
        )

    if not args.skip_ablations and Path(calibrated).exists():
        run(
            [
                PYTHON, "-m", "src.evaluation.run_ablations",
                "--config", "configs/benchmark_edge.yaml",
                "--model", calibrated,
                "--spatial-checkpoint", spatial,
                "--split", args.split, "--role", "test",
                "--out-dir", "results/raw",
                "--tag", f"ablations_{args.split}_seed{args.seed}",
            ],
            "ablations",
        )

    if not args.skip_benchmarks and Path(calibrated).exists():
        run(
            [
                PYTHON, "-m", "src.evaluation.benchmark_latency",
                "--config", "configs/benchmark_edge.yaml",
                "--model", calibrated,
                "--spatial-checkpoint", spatial,
                "--split", args.split, "--role", "test",
                "--device", "cpu",
                "--warmup", "100", "--runs", "1000",
                "--out-dir", "results/raw",
                "--tag", f"latency_{args.split}_seed{args.seed}",
            ],
            "benchmark_latency",
        )
        run(
            [
                PYTHON, "-m", "src.evaluation.benchmark_energy",
                "--config", "configs/benchmark_edge.yaml",
                "--model", calibrated,
                "--spatial-checkpoint", spatial,
                "--split", args.split, "--role", "test",
                "--device", "cpu",
                "--duration-minutes", "3",
                "--compare-frame-skip",
                "--out-dir", "results/raw",
                "--tag", f"energy_{args.split}_seed{args.seed}",
            ],
            "benchmark_energy",
        )

    # The tables and the figures.
    run(
        [PYTHON, "-m", "src.evaluation.aggregate_splits", "--results", "results/raw",
         "--prefix", "video_eval", "--out", "results/tables/summary_over_splits.csv"],
        "aggregate_splits",
    )
    run(
        [PYTHON, "-m", "src.evaluation.make_tables", "--results", "results/raw",
         "--out", "results/tables"],
        "make_tables",
    )
    for script, out in [
        ("plot_roc", "roc.png"),
        ("plot_pr", "pr.png"),
        ("plot_reliability", "reliability.png"),
        ("plot_delay", "delay.png"),
    ]:
        run(
            [PYTHON, "-m", f"src.visualization.{script}", "--results", "results/raw",
             "--out", f"results/figures/{out}"],
            script,
        )

    # The paired statistical tests between the reference model and the
    # temporal model of the split.
    baseline_eval = raw / f"video_eval_spatial_{args.split}_seed{args.seed}.json"
    temporal_eval = raw / f"video_eval_tcn_{args.split}_seed{args.seed}.json"
    if baseline_eval.exists() and temporal_eval.exists():
        for metric in ("detection_delay", "false_alarms_per_hour", "video_recall"):
            run(
                [
                    PYTHON, "-m", "src.evaluation.statistics",
                    "--baseline", str(baseline_eval),
                    "--candidate", str(temporal_eval),
                    "--metric", metric,
                    "--out", f"results/tables/statistics_{metric}_{args.split}.json",
                ],
                f"statistics_{metric}",
            )

    print("reporting finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
