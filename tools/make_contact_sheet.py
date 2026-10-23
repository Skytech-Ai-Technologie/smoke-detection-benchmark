"""Build a labelled contact sheet of a video, used to annotate the smoke onset.

Development helper for the manual onset annotation of the FIRESENSE positive
videos. The output is a single image with ``--columns`` x ``--rows`` tiles,
each labelled with its frame index and timestamp.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def build_sheet(
    video_path: str,
    out_path: str,
    *,
    columns: int = 4,
    rows: int = 4,
    tile_width: int = 320,
    tile_height: int = 240,
    start: int = 0,
    end: int | None = None,
) -> dict:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if end is None:
        end = total
    n_tiles = columns * rows
    indices = np.linspace(start, max(start, end - 1), n_tiles).astype(int)

    frames = {}
    capture = cv2.VideoCapture(video_path)
    wanted = set(int(i) for i in indices)
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame.copy()
        index += 1
    capture.release()

    sheet = np.full((rows * (tile_height + 22), columns * tile_width, 3), 255, dtype=np.uint8)
    for slot, frame_index in enumerate(indices):
        frame = frames.get(int(frame_index))
        if frame is None:
            continue
        resized = cv2.resize(frame, (tile_width, tile_height))
        r, c = divmod(slot, columns)
        y0 = r * (tile_height + 22)
        x0 = c * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = resized
        cv2.putText(
            sheet,
            f"f{int(frame_index)}  t={int(frame_index) / fps:.1f}s",
            (x0 + 5, y0 + tile_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return {"video": video_path, "frames": total, "fps": fps, "indices": [int(i) for i in indices]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    info = build_sheet(
        args.video,
        args.out,
        columns=args.columns,
        rows=args.rows,
        start=args.start,
        end=args.end,
    )
    print(info["frames"], info["fps"], info["indices"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
