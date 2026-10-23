# Calibrated Temporal Smoke Detection

A reproducible, edge-oriented benchmark of smoke detection in video that
compares the classical **Kalman–HSV–CNN** pipeline with fixed temporal rules
and with **calibrated temporal models** (causal TCN, GRU, probabilistic
evidence accumulator).

> **Research question.** *Can calibrated temporal evidence accumulation reduce
> false alarms and detection delay compared with fixed confidence and
> persistence thresholds, while preserving real-time performance and reducing
> energy consumption on edge hardware?*

The repository contains the complete pipeline: data acquisition and
verification, video-level splitting, feature extraction, spatial and temporal
training, calibration, operating-point selection, the four evaluation modes,
the edge benchmarks, the ablation grid, the statistical tests and the
paper-ready tables and figures.

---

## 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.lock.txt
pip install -e .
```

Python 3.10 or 3.11 is required. The authoritative environment is
`requirements.lock.txt`; the hardware and software stack of the reported run
is recorded in `paper/results_summary.md`.

---

## 2. Quick start

```bash
# 1. download and verify the dataset (hashes are checked, nothing is replaced)
python -m src.data.download --config configs/data_firesense.yaml

# 2. inspect the raw videos and build the manifest
python -m src.data.inspect --root data/raw/firesense_smoke
python -m src.data.build_manifest --dataset firesense \
    --root data/raw/firesense_smoke \
    --out data/manifests/firesense_videos.csv

# 3. create the five fixed video-level splits (the test manifests are locked)
python -m src.data.video_split --manifest data/manifests/firesense_videos.csv \
    --out-dir data/manifests --num-splits 5 --seed 42

# 4. cache the candidate features of every video
python -m src.data.extract_frames --manifest data/manifests/firesense_videos.csv \
    --out-dir data/processed/features --annotations data/annotations/onsets.csv

# 5. run the whole experiment for one split
python experiments/run_pipeline.py --split split01 --seed 42
```

Step 5 trains the spatial model, trains the temporal model, calibrates it,
selects the operating point on the validation split and produces the frame,
video and calibration evaluations plus the operating-point sweep.

---

## 3. Repository layout

```
smoke-temporal-detection/
├── configs/                 experiment configurations (one per model)
├── data/
│   ├── annotations/         manual smoke-onset annotations + contact sheets
│   ├── manifests/           video manifests, the five splits, the lock file
│   ├── processed/           cached features (never committed)
│   └── raw/                 downloaded archives (never committed)
├── experiments/
│   ├── run_pipeline.py      one command per split
│   └── registry.csv         machine-readable record of every run
├── paper/                   methods, protocol, dataset card, model card
├── results/
│   ├── figures/             paper figures
│   ├── logs/                per-run logs
│   ├── raw/                 machine-readable results of every experiment
│   └── tables/              paper-ready CSV tables
├── src/
│   ├── data/                download, inspect, manifest, split, extraction
│   ├── models/              front end, spatial CNN, temporal models, detector
│   ├── training/            spatial, temporal, calibration, quantisation
│   ├── evaluation/          frame, video, calibration, cross-dataset, edge
│   ├── visualization/       ROC, PR, reliability, delay, qualitative grids
│   └── utils/               seeding, logging, checkpoints, experiment registry
├── tools/                  result-aggregation and housekeeping scripts
└── tests/                   unit tests, including the video-leakage test
```

---

## 4. The pipeline

```
video decoding
  -> Kalman background model (per-pixel constant-velocity tracker)
  -> HSV low-saturation candidate mask
  -> median filter, connected components, area / aspect-ratio filtering
  -> 50x50 candidate crops
  -> compact spatial CNN (per-crop smoke probability)
  -> calibrated frame probability
  -> temporal module (causal TCN | GRU | evidence accumulator | fixed rule)
  -> adaptive or fixed alarm policy
  -> interpretable decision log
```

### Models

| # | Model | Module | Purpose |
|---|-------|--------|---------|
| 0 | Fixed-rule baseline | `src/models/calibrated_detector.py` | Kalman–HSV–CNN + fixed threshold and persistence |
| 1 | Frame-only CNN | `src/models/spatial_cnn.py` | candidate classifier (318 k parameters) |
| 2 | Fixed temporal voting | `src/models/fixed_temporal.py` | mean / median / majority / max / persistence |
| 3 | Compact GRU | `src/models/temporal_gru.py` | 16-frame window, 32 hidden units |
| 4 | Compact causal TCN | `src/models/temporal_tcn.py` | 16-frame window, 3 dilated layers, 2.5 k parameters |
| 5 | Evidence accumulator | `src/models/evidence_accumulator.py` | non-neural log-odds integration with hysteresis |
| 6 | Calibrated adaptive detector | `src/models/calibrated_detector.py` | the proposed system |

Every decision is logged with the frame probability, the calibrated
probability, the temporal evidence, the model uncertainty, the motion
reliability, the alarm state and the decision, so that an alarm can be
explained after the fact.

---

## 5. Data

| Dataset | Role | Size | Notes |
|---------|------|------|-------|
| **FIRESENSE** (smoke) | primary temporal benchmark | 22 videos, 17 722 frames, 13.9 min | [Zenodo 836749](https://zenodo.org/records/836749), CC BY 4.0 |
| **D-Fire** | spatial pretraining | 21 000+ images | [GitHub](https://github.com/gaia-solutions-on-demand/DFireDataset) — manual acquisition, see below |
| VisiFire / BoWFire | optional external validation | — | only with a verified manifest and licence |

### 5.1 D-Fire cannot be downloaded automatically

The official D-Fire repository links to OneDrive and Kaggle, both of which
require interactive authentication; the OneDrive link is dead and Kaggle
requires an API token. `configs/data_dfire.yaml` therefore contains the
**complete** acquisition and verification procedure, and
`src/data/download.py` records the SHA-256 of whatever archive is found, so
that the manifest hash pins the exact dataset version. The spatial stage
supports D-Fire out of the box (`objective: crop`, region-level YOLO labels);
the reported runs use the FIRESENSE *fire* archive as an additional spatial
source. This deviation is stated explicitly in `paper/dataset_card.md`.

### 5.2 The FIRESENSE videos carry no onset annotation

The public release contains the raw videos only. Two consequences shaped the
whole benchmark and are documented in `paper/dataset_card.md`:

1. **The smoke is already visible in the first frame of 12 of the 13 positive
   videos.** The clips were cut starting from the moment the smoke was
   produced, so there are almost no "pre-onset" frames and the classical
   *detection delay from the annotated onset* degenerates into the *time to
   first alarm*.
2. **There is no region-level smoke annotation**, so the spatial classifier
   cannot be trained with a per-crop label. It is trained with a
   **weakly-supervised multiple-instance objective**: a positive frame must
   contain at least one smoke-like candidate, a negative frame must not
   contain any. The onsets were annotated manually from contact sheets
   (`data/annotations/onsets.csv`, protocol in the dataset card).

### 5.3 Hard negatives

Three of the nine negative videos are genuinely hard: a highway camera with
**truck exhaust plumes** (`testneg06`, `testneg09`) and an **indoor office**
with walking people (`testneg03`). Whether a hard negative falls into the
training, validation or test split of a given split is essentially random
with 22 videos, which is exactly why the benchmark reports the mean and the
standard deviation over **five predefined splits** and never a single split.

---

## 6. Protocol

* **All splits are made at the video level.** Frames of a video never appear
  in two roles; files of one recording session are kept together. The test
  manifests are locked read-only and their hashes are recorded in
  `data/manifests/splits_locked.json`. `tests/test_no_video_leakage.py`
  verifies the property.
* **Training / validation / test / external** are strictly separated. The
  hyper-parameters, the calibration, the operating point and the early
  stopping use the training and validation data only; the test split is
  touched exactly once, for the final reporting.
* **The unit of analysis is the video**, never the frame. Every statistical
  test (paired bootstrap, Wilcoxon signed-rank, McNemar) is performed over
  the per-video outcomes.
* **Every run is recorded** in `experiments/registry.csv` with the git
  commit, the dataset manifest hash, the config hash, the seed, the device,
  the precision, the exact command line and the checkpoint hash. Experiment
  directories are never overwritten.

The complete protocol, including the exact metric definitions, is in
`paper/benchmark_protocol.md`.

---

## 7. Running the benchmark

```bash
# validation
python -m pytest tests/ -q

# data and splits
python -m src.data.inspect --root data/raw
python -m src.data.build_manifest --dataset firesense
python -m src.data.video_split --manifest data/manifests/firesense_videos.csv --num-splits 5
python -m pytest tests/test_no_video_leakage.py

# training
python -m src.training.train_spatial  --config configs/baseline_kalman_hsv_cnn.yaml --seed 42
python -m src.training.train_temporal --config configs/temporal_tcn.yaml  --seed 42
python -m src.training.train_temporal --config configs/temporal_gru.yaml  --seed 42
python -m src.training.train_temporal --config configs/evidence_accumulator.yaml --seed 42
python -m src.training.calibrate --checkpoint CHECKPOINT --method isotonic
python -m src.training.quantize  --checkpoint CHECKPOINT --mode int8

# evaluation
python -m src.evaluation.evaluate_frame      --config configs/benchmark_edge.yaml --model MODEL
python -m src.evaluation.evaluate_video      --config configs/benchmark_edge.yaml --model MODEL
python -m src.evaluation.evaluate_calibration --config configs/benchmark_edge.yaml --model MODEL
python -m src.evaluation.evaluate_cross_dataset --model MODEL --dataset visifire
python -m src.evaluation.run_ablations        --config configs/benchmark_edge.yaml --model MODEL
python -m src.evaluation.sweep_curve          --models spatial=none tcn=MODEL --role test

# edge benchmarks
python -m src.evaluation.benchmark_latency --config configs/benchmark_edge.yaml --model MODEL --warmup 100 --runs 1000
python -m src.evaluation.benchmark_energy  --config configs/benchmark_edge.yaml --model MODEL --duration-minutes 30

# statistics and tables
python -m src.evaluation.statistics --baseline A.json --candidate B.json --metric detection_delay
python -m src.evaluation.make_tables --results results/raw --out results/tables

# figures
python -m src.visualization.plot_roc         --results results/raw
python -m src.visualization.plot_pr          --results results/raw
python -m src.visualization.plot_reliability --results results/raw
python -m src.visualization.plot_delay       --results results/raw
python -m src.visualization.make_qualitative_grid --video VIDEO --model MODEL
```

---

## 8. Results

See `paper/results_summary.md` for the numbers, `results/tables/` for the
machine-readable tables and `results/figures/` for the figures.

---

## 9. Reproducibility

`src/utils/experiment_registry.py` writes one record per run:

```json
{
  "experiment_id": "tcn_split01_seed42",
  "git_commit": "...",
  "dataset_manifest_sha256": "...",
  "config_sha256": "...",
  "seed": 42,
  "device": "cuda:0:NVIDIA GeForce RTX 5070 Laptop GPU",
  "precision": "fp32",
  "command": "python -m src.training.train_temporal --config configs/temporal_tcn.yaml",
  "checkpoint": "results/raw/tcn_split01_seed42/model.pt",
  "checkpoint_sha256": "..."
}
```

The dataset archives are verified by MD5 **and** SHA-256; a changed upstream
archive aborts the download instead of silently replacing the data.

---

## 10. Licence

The code is released under the Apache-2.0 licence (`LICENSE`). The FIRESENSE
videos are CC BY 4.0 and must be cited accordingly; the D-Fire images and
labels keep their own licence. No dataset is redistributed in this
repository.
