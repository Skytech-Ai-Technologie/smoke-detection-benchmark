"""Edge latency benchmark.

The benchmark measures the **end-to-end** cost of the pipeline and reports the
cost of every stage separately:

    video decoding
    -> Kalman background model
    -> HSV processing and candidate extraction
    -> spatial CNN
    -> temporal model
    -> alarm policy
    -> logging / output

Reporting only the CNN inference speed would overstate the real-time
performance of the system by more than an order of magnitude, so the reported
"system FPS" is the throughput of the complete pipeline.

Example
-------
    python -m src.evaluation.benchmark_latency \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn/model.pt \\
        --spatial-checkpoint results/raw/spatial/model.pt \\
        --warmup 100 --runs 1000
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.experiment_registry import append_to_registry, build_record, experiment_id
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .common import add_common_arguments, resolve_config, write_result
from .runner import load_specs_from_manifest

__all__ = ["percentiles", "benchmark_stream", "main"]

LOGGER = get_logger("evaluation.benchmark_latency")


def percentiles(values: np.ndarray) -> dict[str, float]:
    """Latency percentiles (milliseconds) and jitter."""
    if values.size == 0:
        return {}
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "min_ms": float(values.min()),
        # Jitter: mean absolute deviation of the frame period around its mean.
        "jitter_ms": float(statistics.mean(abs(v - values.mean()) for v in values)),
    }


def benchmark_stream(
    video_path: str,
    *,
    spatial_checkpoint: str | None,
    temporal_checkpoint: str | None,
    device: str,
    warmup: int,
    runs: int,
    policy_overrides: dict[str, Any] | None = None,
    fps: float = 25.0,
) -> dict[str, Any]:
    """Run the pipeline over one video and collect the latency statistics."""
    import cv2
    import torch

    from ..models.builder import build_detector

    bundle = build_detector(
        spatial_checkpoint,
        temporal_checkpoint,
        device=device,
        policy_overrides=policy_overrides,
        fps=fps,
    )
    detector = bundle.detector

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    frames: list[np.ndarray] = []
    while len(frames) < warmup + runs:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError(f"could not re-open video: {video_path}")
            detector.reset("bench")
            continue
        frames.append(frame)
    capture.release()

    # Warm-up (kernel autotuning, allocator, first-touch costs).
    for i in range(warmup):
        detector.process_frame(frames[i], i, timestamp=i / fps)
    detector.reset("bench")
    for key in detector.stage_times:
        detector.stage_times[key] = 0.0

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    latencies: list[float] = []
    wall_start = time.perf_counter()
    for i in range(runs):
        t0 = time.perf_counter()
        detector.process_frame(frames[warmup + i], warmup + i, timestamp=(warmup + i) / fps)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - wall_start

    stats = percentiles(np.asarray(latencies))
    frame_budget_ms = 1000.0 / fps if fps > 0 else 0.0
    dropped = int(np.sum(np.asarray(latencies) > frame_budget_ms)) if frame_budget_ms else 0

    peak_ram_mb = 0.0
    try:
        import psutil

        peak_ram_mb = psutil.Process().memory_info().rss / (1024**2)
    except Exception:  # pragma: no cover
        pass

    peak_vram_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    stage_ms = {k: v * 1000.0 / max(runs, 1) for k, v in detector.stage_times.items()}
    stage_share = {
        k: (v / sum(stage_ms.values()) if sum(stage_ms.values()) else 0.0) for k, v in stage_ms.items()
    }

    return {
        "video": str(video_path),
        "n_frames": runs,
        "fps_nominal": fps,
        "system_fps": runs / wall if wall > 0 else 0.0,
        "wall_time_s": wall,
        "frame_budget_ms": frame_budget_ms,
        "dropped_frames": dropped,
        "dropped_frame_ratio": dropped / runs if runs else 0.0,
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
        "stage_ms_per_frame": stage_ms,
        "stage_share": stage_share,
        **stats,
    }


def _concurrent_benchmark(
    video_path: str,
    *,
    streams: int,
    **kwargs,
) -> dict[str, Any]:
    """Run several independent detector instances on the same stream."""
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker() -> None:
        result = benchmark_stream(video_path, **kwargs)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(streams)]
    wall_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall_start

    total_frames = sum(r["n_frames"] for r in results)
    return {
        "streams": streams,
        "aggregate_fps": total_frames / wall if wall > 0 else 0.0,
        "per_stream_fps": [r["system_fps"] for r in results],
        "p95_ms_per_stream": [r["p95_ms"] for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Edge latency benchmark.")
    add_common_arguments(parser)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--video", default=None, help="video to benchmark (default: first test video)")
    parser.add_argument("--concurrent-streams", type=int, nargs="*", default=None)
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)

    edge = config.get("benchmark", {}).get("edge", {})
    warmup = args.warmup if args.warmup is not None else int(edge.get("warmup_frames", 100))
    runs = args.runs if args.runs is not None else int(edge.get("benchmark_frames", 1000))

    if args.video:
        video_path = args.video
    else:
        specs = load_specs_from_manifest(
            args.manifest,
            args.feature_dir,
            role=args.role,
            split=args.split,
            splits_file=args.splits_file,
        )
        if not specs:
            raise RuntimeError("no videos selected for the benchmark")
        video_path = specs[0].path

    LOGGER.info("benchmarking %s (warmup=%d, runs=%d)", video_path, warmup, runs)

    result = benchmark_stream(
        video_path,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        warmup=warmup,
        runs=runs,
        policy_overrides=args.policy_overrides,
    )

    streams = args.concurrent_streams
    if streams is None:
        streams = edge.get("concurrent_streams", [1])
    concurrency = []
    for count in streams:
        if int(count) <= 1:
            continue
        LOGGER.info("benchmarking with %s concurrent streams", count)
        concurrency.append(
            _concurrent_benchmark(
                video_path,
                streams=int(count),
                spatial_checkpoint=args.spatial_checkpoint,
                temporal_checkpoint=args.model,
                device=args.device,
                warmup=max(10, warmup // 10),
                runs=max(100, runs // 2),
                policy_overrides=args.policy_overrides,
            )
        )

    payload = {
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "single_stream": result,
        "concurrency": concurrency,
    }

    name = args.tag or experiment_id("latency", split=args.split, seed=args.seed)
    write_result(args.out_dir, name, payload)

    csv_path = Path(args.out_dir) / f"{name}_stages.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "ms_per_frame", "share"])
        for stage, value in result["stage_ms_per_frame"].items():
            writer.writerow([stage, f"{value:.4f}", f"{result['stage_share'][stage]:.4f}"])

    record = build_record(
        exp_id=name,
        config={"warmup": warmup, "runs": runs},
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        model="latency_benchmark",
        notes="edge latency benchmark",
        metrics={
            "system_fps": result["system_fps"],
            "p50_ms": result["p50_ms"],
            "p95_ms": result["p95_ms"],
            "p99_ms": result["p99_ms"],
            "dropped_frames": result["dropped_frames"],
        },
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "system FPS=%.2f | p50=%.2f ms p95=%.2f ms p99=%.2f ms | dropped=%d",
        result["system_fps"],
        result["p50_ms"],
        result["p95_ms"],
        result["p99_ms"],
        result["dropped_frames"],
    )
    for stage, value in result["stage_ms_per_frame"].items():
        LOGGER.info("  %-16s %8.3f ms (%4.1f%%)", stage, value, 100 * result["stage_share"][stage])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
