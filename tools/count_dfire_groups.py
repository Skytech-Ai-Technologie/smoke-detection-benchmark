"""Count the DFire test-set groups (smoke / fire_only / none) exactly once."""
import sys
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", buffering=1)
labels_dir = Path(r"D:\FORFIRE CODES\forfire1\wildfire-fusion\data\dfire\labels\test")
counts = {"smoke": 0, "fire_only": 0, "none": 0}
for label_path in labels_dir.glob("*.txt"):
    classes = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5:
            classes.add(int(parts[0]))
    if 0 in classes:
        counts["smoke"] += 1
    elif 1 in classes:
        counts["fire_only"] += 1
    else:
        counts["none"] += 1
print(counts, "total", sum(counts.values()))
