"""Edge energy benchmark.

The benchmark reports

* the CPU time consumed per processed frame (measured),
* the energy per processed frame (estimated from the measured CPU time and a
  configurable effective power, or integrated from the measured GPU power
  draw when a GPU is used),
* the device temperature at the requested checkpoints (5 / 15 / 30 minutes),
* the same quantities for the frame-skipping variant, so that the energy
  saving of the adaptive policy can be quantified.

The energy figure is an *estimate*: a laptop has no external power meter, so
the CPU time is converted to joules with the effective power of the compute
device, which must be reported alongside the number. When a GPU is used, the
package power is sampled with NVML and integrated over the wall-clock time,
which is a direct measurement of the accelerator's contribution.

Example
-------
    python -m src.evaluation.benchmark_energy \\
        --config configs/benchmark_edge.yaml \\
        --model results/raw/tcn/model.pt \\
        --duration-minutes 30
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
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

__all__ = ["EnergyMonitor", "benchmark_energy", "main"]

LOGGER = get_logger("evaluation.benchmark_energy")


class EnergyMonitor:
    """Samples the process CPU time, the GPU power draw and the temperature."""

    def __init__(self, device: str = "cpu", sample_interval: float = 1.0) -> None:
        self.device = device
        self.sample_interval = float(sample_interval)
        self.samples: list[dict[str, float]] = []
        self.temperatures: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._process = None
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ gpu
    @staticmethod
    def _nvidia_smi() -> dict[str, float] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            parts = result.stdout.strip().split(",")
            if len(parts) < 2:
                return None
            return {"power_w": float(parts[0]), "temperature_c": float(parts[1])}
        except Exception:
            return None

    def _cpu_temperature(self) -> float | None:
        try:
            import psutil

            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            if not temps:
                return None
            best = None
            for name in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                if name in temps and temps[name]:
                    return float(temps[name][0].current)
            for values in temps.values():
                if values:
                    best = float(values[0].current)
                    break
            return best
        except Exception:
            return None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._start_wall = time.perf_counter()
        self._start_cpu = self._cpu_seconds()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _cpu_seconds(self) -> float:
        if self._process is None:
            return time.process_time()
        try:
            times = self._process.cpu_times()
            return float(times.user + times.system)
        except Exception:  # pragma: no cover
            return time.process_time()

    def _sampler(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, float] = {
                "t": time.perf_counter() - self._start_wall,
                "cpu_s": self._cpu_seconds() - self._start_cpu,
            }
            gpu = self._nvidia_smi() if self.device.startswith("cuda") else None
            if gpu:
                sample["gpu_power_w"] = gpu["power_w"]
                sample["gpu_temperature_c"] = gpu["temperature_c"]
                self.temperatures.append((sample["t"], gpu["temperature_c"]))
            cpu_temp = self._cpu_temperature()
            if cpu_temp is not None:
                sample["cpu_temperature_c"] = cpu_temp
            self.samples.append(sample)
            self._stop.wait(self.sample_interval)

    # ---------------------------------------------------------------- report
    def report(self, n_frames: int, effective_power_w: float) -> dict[str, Any]:
        wall = time.perf_counter() - self._start_wall
        cpu_s = self._cpu_seconds() - self._start_cpu
        out: dict[str, Any] = {
            "wall_time_s": wall,
            "cpu_time_s": cpu_s,
            "n_frames": n_frames,
            "cpu_ms_per_frame": 1000.0 * cpu_s / n_frames if n_frames else float("nan"),
            "effective_power_watts": effective_power_w,
            "estimated_energy_j_per_frame": (
                cpu_s * effective_power_w / n_frames if n_frames else float("nan")
            ),
        }
        if self.samples:
            powers = [s["gpu_power_w"] for s in self.samples if "gpu_power_w" in s]
            if powers:
                mean_power = float(np.mean(powers))
                out["gpu_mean_power_w"] = mean_power
                out["gpu_peak_power_w"] = float(np.max(powers))
                out["gpu_energy_j_per_frame"] = (
                    mean_power * wall / n_frames if n_frames else float("nan")
                )
                out["gpu_estimated_energy_j_per_frame"] = (
                    mean_power * wall / n_frames if n_frames else float("nan")
                )
            cpu_temps = [s["cpu_temperature_c"] for s in self.samples if "cpu_temperature_c" in s]
            if cpu_temps:
                out["cpu_temperature_start_c"] = float(cpu_temps[0])
                out["cpu_temperature_end_c"] = float(cpu_temps[-1])
                out["cpu_temperature_max_c"] = float(np.max(cpu_temps))
            gpu_temps = [s["gpu_temperature_c"] for s in self.samples if "gpu_temperature_c" in s]
            if gpu_temps:
                out["gpu_temperature_start_c"] = float(gpu_temps[0])
                out["gpu_temperature_end_c"] = float(gpu_temps[-1])
                out["gpu_temperature_max_c"] = float(np.max(gpu_temps))
        return out


def benchmark_energy(
    video_path: str,
    *,
    spatial_checkpoint: str | None,
    temporal_checkpoint: str | None,
    device: str,
    duration_minutes: float,
    policy_overrides: dict[str, Any] | None = None,
    effective_power_w: float = 15.0,
    fps: float = 25.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the pipeline continuously and measure its energy consumption."""
    import cv2

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

    monitor = EnergyMonitor(device=device)
    monitor.start()

    n_frames = 0
    deadline = time.perf_counter() + duration_minutes * 60.0
    index = 0
    try:
        while time.perf_counter() < deadline:
            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    break
                detector.reset("energy")
                index = 0
                continue
            detector.process_frame(frame, index, timestamp=index / fps)
            n_frames += 1
            index += 1
            if max_frames is not None and n_frames >= max_frames:
                break
    finally:
        monitor.stop()
        capture.release()

    report = monitor.report(n_frames, effective_power_w)
    report["video"] = str(video_path)
    report["skipped_frames"] = detector._skipped_total
    report["skip_ratio"] = detector._skipped_total / n_frames if n_frames else 0.0
    report["stage_ms_per_frame"] = {
        k: v * 1000.0 / max(n_frames, 1) for k, v in detector.stage_times.items()
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Edge energy benchmark.")
    add_common_arguments(parser)
    parser.add_argument("--duration-minutes", type=float, default=5.0)
    parser.add_argument("--effective-power-watts", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--compare-frame-skip", action="store_true", help="also benchmark the frame-skipping variant")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    set_seed(args.seed)
    config = resolve_config(args)

    edge = config.get("benchmark", {}).get("edge", {})
    effective_power = (
        args.effective_power_watts
        if args.effective_power_watts is not None
        else float(edge.get("effective_power_watts", 15.0))
    )

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

    LOGGER.info("energy benchmark on %s for %.1f min", video_path, args.duration_minutes)
    baseline = benchmark_energy(
        video_path,
        spatial_checkpoint=args.spatial_checkpoint,
        temporal_checkpoint=args.model,
        device=args.device,
        duration_minutes=args.duration_minutes,
        policy_overrides=args.policy_overrides,
        effective_power_w=effective_power,
        max_frames=args.max_frames,
    )

    payload: dict[str, Any] = {
        "model": args.model,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "effective_power_watts": effective_power,
        "baseline": baseline,
    }

    if args.compare_frame_skip:
        skip_overrides = dict(args.policy_overrides)
        skip_overrides["frame_skip"] = True
        skip_overrides.setdefault("frame_skip_interval", 2)
        LOGGER.info("energy benchmark with adaptive frame skipping")
        skipping = benchmark_energy(
            video_path,
            spatial_checkpoint=args.spatial_checkpoint,
            temporal_checkpoint=args.model,
            device=args.device,
            duration_minutes=args.duration_minutes,
            policy_overrides=skip_overrides,
            effective_power_w=effective_power,
            max_frames=args.max_frames,
        )
        payload["frame_skipping"] = skipping
        base_j = baseline.get("estimated_energy_j_per_frame", float("nan"))
        skip_j = skipping.get("estimated_energy_j_per_frame", float("nan"))
        if base_j and skip_j == skip_j:
            payload["energy_reduction_percent"] = 100.0 * (1.0 - skip_j / base_j)

    name = args.tag or experiment_id("energy", split=args.split, seed=args.seed)
    write_result(args.out_dir, name, payload)

    record = build_record(
        exp_id=name,
        config={"duration_minutes": args.duration_minutes, "effective_power_watts": effective_power},
        seed=args.seed,
        device=args.device,
        precision="fp32",
        command=" ".join(sys.argv),
        checkpoint=args.model,
        model="energy_benchmark",
        notes="edge energy benchmark",
        metrics={
            "cpu_ms_per_frame": baseline.get("cpu_ms_per_frame"),
            "estimated_energy_j_per_frame": baseline.get("estimated_energy_j_per_frame"),
        },
    )
    append_to_registry(args.registry, record)

    LOGGER.info(
        "energy | cpu %.2f ms/frame | %.3f J/frame (estimated at %.1f W) | %d frames",
        baseline.get("cpu_ms_per_frame", float("nan")),
        baseline.get("estimated_energy_j_per_frame", float("nan")),
        effective_power,
        baseline.get("n_frames", 0),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
