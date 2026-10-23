"""Rename the result files so that the model name is explicit in the file name."""

import glob
import os
import sys

BASE = os.path.abspath("results/raw")
PREFIXES = ("video_eval", "frame_eval", "calibration_eval", "thresholds", "sweep")


def main() -> int:
    renamed = 0
    for path in glob.glob(os.path.join(BASE, "*.json")):
        name = os.path.basename(path)
        for prefix in PREFIXES:
            if not name.startswith(prefix + "_") or "_split0" not in name:
                continue
            rest = name[len(prefix) + 1 :]
            if rest.startswith("split0"):
                target = os.path.join(BASE, f"{prefix}_tcn_{rest}")
                if os.path.exists(target):
                    os.remove(path)
                else:
                    os.rename(path, target)
                    renamed += 1
                print("renamed", name, "->", os.path.basename(target))
            break
    print(f"{renamed} files renamed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
