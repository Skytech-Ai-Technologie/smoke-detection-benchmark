# Data directory

```
data/
├── raw/                  downloaded archives and the extracted videos (never committed)
│   ├── downloads/        the original archives + the download metadata JSON
│   └── firesense_smoke/  the extracted FIRESENSE smoke videos
├── annotations/          the manual smoke-onset annotations and the contact sheets
├── manifests/            the video manifests, the splits and the lock file
└── processed/
    └── features/         the cached candidate features of every video
```

## `data/raw/`

Raw data is **never committed**. Obtain it with

```bash
python -m src.data.download --config configs/data_firesense.yaml
```

The downloader verifies the size, the MD5 and the SHA-256 of every archive
and refuses to replace an existing file whose hash differs. A changed
upstream archive is a *dataset version change* and has to be handled
explicitly (`--overwrite`), because the manifest hash pins the dataset
version that produced the results.

`data/raw/downloads/<dataset>_metadata.json` records the URLs, the sizes,
the hashes, the download date, the licence and the citation.

## `data/annotations/`

`onsets.csv` holds the **manual smoke-onset annotations** of the FIRESENSE
positive videos. The annotation protocol is documented in
`paper/dataset_card.md`; the contact sheets that were used for the
annotation are kept in `sheets/` (whole video) and `sheets_fine/` (first 120
frames). The burned-in cyan smoke-region contours of the original recordings
are detected by `tools/detect_annotations.py` and recorded as a cross-check.

## `data/manifests/`

| File | Content |
|---|---|
| `firesense_videos.csv` | one row per video: id, path, label, frames, fps, resolution, duration, SHA-256, session, onset, source URL |
| `splits.csv` | every (split, role, video) triple of the five splits |
| `split0N_{train,validation,test}.csv` | the per-split role manifests (the test files are read-only) |
| `train.csv`, `validation.csv`, `test.csv` | convenience copies of the primary split |
| `lovo.csv` | the leave-one-video-out registry |
| `splits_locked.json` | the SHA-256 of every split file at the moment of locking |

The test manifests are **read-only** after the splits are finalised. The
lock is removed and re-applied automatically when the splits are rebuilt.

## `data/processed/features/`

One `*.npz` per video with the cached output of the front end:

| Array | Shape | Content |
|---|---|---|
| `frame_brightness`, `frame_blur`, `frame_fg_ratio`, `frame_motion`, `frame_max_area_ratio` | (T,) | per-frame statistics |
| `frame_label` | (T,) | 1 from the smoke onset onwards in a positive video |
| `cand_frame_index` | (N,) | the frame of every candidate |
| `cand_bbox`, `cand_area`, `cand_hsv`, `cand_motion` | (N, ...) | the candidate metadata |
| `cand_crop` | (N, 50, 50, 3) | the crops used by the spatial CNN |
| `cand_label` | (N,) | the three-valued crop label (-1 = ignore) |
| `onset_frame`, `onset_source` | scalar / string | the resolved onset and its origin |

The cache is what makes the experiments affordable: the front end is the
expensive part of the pipeline and it does not depend on the annotations, so
the labels can be refreshed with

```bash
python -m src.data.extract_frames --manifest data/manifests/firesense_videos.csv \
    --out-dir data/processed/features --annotations data/annotations/onsets.csv --relabel-only
```

without re-running the Kalman model and the candidate extraction.
