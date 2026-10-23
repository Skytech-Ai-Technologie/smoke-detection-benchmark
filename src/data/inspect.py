"""Inspect the raw data directory and report video metadata.

The inspector produces the metadata that the manifest needs (frame count, FPS,
resolution, duration, codec) and prints a human-readable summary. It never
modifies the raw data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

from ..utils.logging import get_logger, setup_logging

__all__ = ["video_metadata", "iter_videos", "main", "VIDEO_EXTENSIONS"]

LOGGER = get_logger("data.inspect")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}


def iter_videos(root: str | Path) -> list[Path]:
    """Return all video files under ``root`` in a deterministic order."""
    base = Path(root)
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def video_metadata(path: str | Path) -> dict[str, Any]:
    """Extract technical metadata from a video file."""
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return {"path": str(source), "error": "could not open video"}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    capture.release()

    codec = ""
    if fourcc:
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()

    duration = frames / fps if fps > 0 else 0.0
    return {
        "path": str(source),
        "num_frames": frames,
        "fps": round(fps, 4),
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "codec": codec,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Inspect raw video data.")
    parser.add_argument("--root", default="data/raw", help="root directory of the raw data")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    videos = iter_videos(args.root)
    LOGGER.info("found %d video files under %s", len(videos), args.root)

    records = [video_metadata(path) for path in videos]
    total_frames = sum(r.get("num_frames", 0) for r in records)
    total_seconds = sum(r.get("duration_sec", 0.0) for r in records)
    LOGGER.info("total: %d frames, %.1f s (%.2f h)", total_frames, total_seconds, total_seconds / 3600)

    for record in records:
        LOGGER.info(
            "%-60s %5d frames  %6.2f fps  %4dx%-4d  %s",
            Path(record["path"]).name,
            record.get("num_frames", 0),
            record.get("fps", 0.0),
            record.get("width", 0),
            record.get("height", 0),
            record.get("codec", ""),
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
