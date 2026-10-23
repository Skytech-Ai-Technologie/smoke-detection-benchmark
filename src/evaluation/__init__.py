"""Evaluation: frame, video, calibration, cross-dataset and edge benchmarks."""

from .benchmark_energy import EnergyMonitor, benchmark_energy
from .benchmark_latency import benchmark_stream, percentiles
from .common import add_common_arguments, resolve_config, write_result
from .evaluate_cross_dataset import EXTERNAL_DATASETS
from .metrics import (
    aggregate_video_metrics,
    calibration_metrics,
    evaluate_video_events,
    frame_metrics,
    mcnemar_test,
    paired_bootstrap,
    wilcoxon_signed_rank,
)
from .runner import RunResult, VideoSpec, load_specs_from_manifest, run_detector

__all__ = [
    "EnergyMonitor",
    "benchmark_energy",
    "benchmark_stream",
    "percentiles",
    "add_common_arguments",
    "resolve_config",
    "write_result",
    "EXTERNAL_DATASETS",
    "aggregate_video_metrics",
    "calibration_metrics",
    "evaluate_video_events",
    "frame_metrics",
    "mcnemar_test",
    "paired_bootstrap",
    "wilcoxon_signed_rank",
    "RunResult",
    "VideoSpec",
    "load_specs_from_manifest",
    "run_detector",
]
