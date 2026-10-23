"""Ad-hoc analysis of the cached FIRESENSE features (development helper)."""

import glob
import sys

import numpy as np

rows = []
for path in sorted(glob.glob("data/processed/features/*.npz")):
    with np.load(path) as d:
        vid = str(d["video_id"])
        lab = str(d["label"])
        a = d["frame_max_area_ratio"]
        m = d["frame_motion"]
        fg = d["frame_fg_ratio"]
        cf = d["cand_frame_index"]
        n = len(a)
        ncc = np.bincount(cf, minlength=n) if len(cf) else np.zeros(n, int)
        # fraction of frames with at least one candidate
        frac_with = float((ncc > 0).mean())
        # largest contiguous run of frames with a candidate
        run = best = 0
        for v in ncc:
            run = run + 1 if v > 0 else 0
            best = max(best, run)
        rows.append(
            (
                vid,
                lab,
                n,
                float(a.max()),
                float(np.percentile(a, 95)),
                float(a.mean()),
                int((a > 0.01).sum()),
                int(ncc.sum()),
                float(m.mean()),
                float(fg.mean()),
                frac_with,
                best,
            )
        )

header = (
    f"{'video':28s} {'label':9s} {'n':>5s} {'maxAR':>7s} {'p95AR':>7s} {'meanAR':>7s} "
    f"{'#>1%':>5s} {'#cand':>6s} {'motion':>7s} {'fgrat':>6s} {'fracC':>6s} {'maxrun':>6s}"
)
print(header)
for r in rows:
    print(
        f"{r[0]:28s} {r[1]:9s} {r[2]:5d} {r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f} "
        f"{r[6]:5d} {r[7]:6d} {r[8]:7.2f} {r[9]:6.3f} {r[10]:6.3f} {r[11]:6d}"
    )

print()
print("negatives: max of frame_max_area_ratio per video")
neg = [r[3] for r in rows if r[1] == "negative"]
pos = [r[3] for r in rows if r[1] == "smoke"]
print("  neg maxAR values:", sorted(round(v, 4) for v in neg))
print("  pos maxAR values:", sorted(round(v, 4) for v in pos))
print("  neg max run with candidate:", sorted(r[11] for r in rows if r[1] == "negative"))
print("  pos max run with candidate:", sorted(r[11] for r in rows if r[1] == "smoke"))
