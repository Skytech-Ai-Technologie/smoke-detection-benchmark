"""Logging helpers: consistent console + file logging for every module."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["get_logger", "setup_logging", "Timer", "log_environment"]

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO, logfile: str | os.PathLike[str] | None = None) -> None:
    """Install the root handlers. Safe to call more than once."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        if logfile is not None:
            _add_file_handler(root, Path(logfile))
        return

    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(console)

    if logfile is not None:
        _add_file_handler(root, Path(logfile))

    _CONFIGURED = True


def _add_file_handler(root: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root logger on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


@contextmanager
def Timer(name: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Time a block and log the duration.

    Yields a dict so the caller can record ``elapsed`` programmatically.
    """
    log = logger or get_logger("timer")
    stats: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["elapsed"] = time.perf_counter() - start
        log.info("%s took %.3f s", name, stats["elapsed"])


def log_environment(logger: logging.Logger | None = None) -> dict[str, object]:
    """Log and return a description of the runtime environment.

    The result is embedded in every experiment record so that a run can be
    traced back to the exact software and hardware stack that produced it.
    """
    import platform
    import sys

    log = logger or get_logger("environment")
    info: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import cv2

        info["opencv_version"] = cv2.__version__
        info["opencv_build_threads"] = cv2.getNumThreads()
    except Exception:  # pragma: no cover
        pass

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_mb"] = int(props.total_memory // (1024**2))
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        pass

    try:
        import psutil

        info["cpu_model"] = _cpu_model_name()
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:  # pragma: no cover
        pass

    for key, value in info.items():
        log.info("%-24s %s", key, value)
    return info


def _cpu_model_name() -> str:
    """Best-effort CPU model string (Windows / Linux / macOS)."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class _CPUInfo(ctypes.Structure):
                _fields_ = [
                    ("ProcessorNameString", ctypes.c_char * 256),
                    ("PhysicalMemory", ctypes.c_ulonglong),
                ]

            info = _CPUInfo()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(info))
            return info.ProcessorNameString.decode(errors="ignore").strip()
        if system == "Linux":
            out = subprocess.check_output(
                "lscpu | grep 'Model name' | head -1", shell=True, text=True
            )
            return out.split(":", 1)[-1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"
