"""Report the composition of every split.

The composition of a split determines how much signal each role receives.
With 22 videos and three "hard" negative videos (the traffic and indoor
recordings), the amount of negative evidence that lands in the training
split varies enormously between the splits. This script quantifies it.

Usage:
    python tools/split_composition.py
"""

import csv
import glob
import os
import sys

import numpy as np


def load_splits(path="data/manifests/splits.csv"):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def video_stats():
    stats = {}
    for path in sorted(glob.glob("data/processed/features/*.npz")):
        with np.load(path) as d:
            video_id = str(d["video_id"])
            stats[video_id] = {
                "label": str(d["label"]),
                "frames": int(len(d["frame_brightness"])),
                "candidate_frames": int(len(set(d["cand_frame_index"].tolist()))),
                "candidates": int(len(d["cand_frame_index"])),
            }
    return stats


def main() -> int:
    splits = load_splits()
    stats = video_stats()

    by_split = {}
    for row in splits:
        by_split.setdefault(row["split_id"], []).append(row)

    header = (
        f"{'split':8s} {'role':11s} {'nvid':>5s} {'pos':>4s} {'neg':>4s} "
        f"{'negCandFrames':>14s} {'negCandVideos':>14s} {'frames':>9s}"
    )
    print(header)
    print("-" * len(header))
    for split_id in sorted(by_split):
        for role in ("train", "validation", "test"):
            rows = [r for r in by_split[split_id] if r["role"] == role]
            neg = [r for r in rows if stats.get(r["video_id"], {}).get("label") != "smoke"]
            neg_cand_frames = sum(stats.get(r["video_id"], {}).get("candidate_frames", 0) for r in neg)
            neg_cand_videos = sum(
                1 for r in neg if stats.get(r["video_id"], {}).get("candidate_frames", 0) > 0
            )
            frames = sum(stats.get(r["video_id"], {}).get("frames", 0) for r in rows)
            n_pos = sum(1 for r in rows if stats.get(r["video_id"], {}).get("label") == "smoke")
            print(
                f"{split_id:8s} {role:11s} {len(rows):5d} {n_pos:4d} {len(neg):4d} "
                f"{neg_cand_frames:14d} {neg_cand_videos:14d} {frames:9d}"
            )
        print()

    print("per-video candidate statistics")
    print(f"{'video':30s} {'label':9s} {'frames':>7s} {'candFrames':>11s} {'cands':>7s}")
    for video_id, info in sorted(stats.items(), key=lambda kv: (kv[1]["label"], kv[0])):
        print(
            f"{video_id:30s} {info['label']:9s} {info['frames']:7d} "
            f"{info['candidate_frames']:11d} {info['candidates']:7d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
