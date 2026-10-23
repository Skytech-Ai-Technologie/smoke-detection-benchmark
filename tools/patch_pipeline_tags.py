"""Patch the tag names of the pipeline so that the model is in the file name."""

import sys

PATH = "experiments/run_pipeline.py"
PREFIXES = ("frame_eval", "video_eval", "calibration_eval", "thresholds", "sweep")


def main() -> int:
    with open(PATH, encoding="utf-8") as handle:
        source = handle.read()
    for prefix in PREFIXES:
        source = source.replace(
            f'f"{prefix}_{{args.split}}_seed{{args.seed}}"',
            f'f"{prefix}_tcn_{{args.split}}_seed{{args.seed}}"',
        )
    source = source.replace(
        'f"thresholds_{args.split}_seed{args.seed}.json"',
        'f"thresholds_tcn_{args.split}_seed{args.seed}.json"',
    )
    with open(PATH, "w", encoding="utf-8") as handle:
        handle.write(source)
    print("patched", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
