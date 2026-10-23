"""Detect the burned-in smoke-region annotations of the FIRESENSE videos.

The positive FIRESENSE recordings contain a cyan contour around the smoke
region. Counting the cyan pixels per frame yields the *official* onset
annotation, which is far more reliable than any heuristic.

Usage:
    python tools/detect_annotations.py --root data/raw/firesense_smoke/pos
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def cyan_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return the mask of the cyan annotation pixels of a frame."""
    b = frame_bgr[..., 0].astype(np.int16)
    g = frame_bgr[..., 1].astype(np.int16)
    r = frame_bgr[..., 2].astype(np.int16)
    # Cyan: high green and blue, low red.
    return ((g > 120) & (b > 120) & (r < 100) & (g - r > 60) & (b - r > 60)).astype(np.uint8)


def analyse(path: str, min_pixels: int = 20) -> dict:
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    counts: list[int] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        counts.append(int(cyan_mask(frame).sum()))
        index += 1
    capture.release()
    counts_arr = np.asarray(counts)
    annotated = counts_arr >= min_pixels
    onset = int(np.argmax(annotated)) if annotated.any() else None
    return {
        "video": os.path.basename(path),
        "frames": index,
        "fps": float(fps),
        "onset_frame": onset,
        "n_annotated_frames": int(annotated.sum()),
        "max_cyan_pixels": int(counts_arr.max()) if counts_arr.size else 0,
        "first_counts": counts_arr[:40].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/firesense_smoke/pos")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--out", default="data/annotations/onsets.json")
    args = parser.parse_args()

    results = []
    for path in sorted(glob.glob(os.path.join(args.root, "*.avi"))):
        info = analyse(path, args.min_pixels)
        results.append(info)
        print(
            f"{info['video']:24s} frames={info['frames']:5d} onset={str(info['onset_frame']):>6s} "
            f"annotated={info['n_annotated_frames']:5d} maxcyan={info['max_cyan_pixels']:6d}"
        )

    import json

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
