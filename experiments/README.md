# Experiments

## One command per split

```bash
python experiments/run_pipeline.py --split split01 --seed 42
```

The pipeline performs, in order:

1. **Stage B** - spatial training (weakly-supervised MIL objective on
   FIRESENSE, region-level BCE on D-Fire) with the number of epochs selected
   by cross-validation inside the training split,
2. **Stage C** - temporal training of the causal TCN on the training videos,
   using the out-of-fold spatial logits,
3. **Stage D** - calibration of the temporal model on the validation split,
   comparing none / temperature / Platt / isotonic,
4. **operating-point selection** on the validation split,
5. **evaluation** at the frame level, the video level and the calibration
   level,
6. **the operating-point sweep** that produces the false-alarms-per-hour
   versus recall curve.

The logs are written to `results/logs/<split>_seed<seed>/` and the results to
`results/raw/`.

## The experiment registry

`experiments/registry.csv` has one row per run:

| Column | Content |
|---|---|
| `experiment_id` | unique identifier, e.g. `tcn_split01_seed42_fp16` |
| `timestamp_utc` | ISO timestamp |
| `git_commit` | the commit of the working tree (or `unknown`) |
| `dataset_manifest_sha256` | hash of the split registry used by the run |
| `config_sha256` | hash of the canonical JSON of the configuration |
| `seed` | the random seed |
| `device` | the compute device |
| `precision` | fp32 / fp16 / int8 |
| `command` | the exact command line |
| `checkpoint` | the path of the artefact |
| `checkpoint_sha256` | the hash of the artefact |
| `split`, `model`, `notes` | the context of the run |
| `metrics_json` | the metrics of the run |

An experiment directory is **never overwritten**: when the target directory
already contains files, a unique suffix is appended and the new directory is
used.

## The five splits

```bash
for s in split01 split02 split03 split04 split05; do
    python experiments/run_pipeline.py --split $s --seed 42
done
```

The splits are independent (not nested), so the five runs give five
independent estimates of the same quantity. The paper reports the mean and
the standard deviation over the splits, and every statistical test is paired
over the videos.

## Ablations

```bash
python -m src.evaluation.run_ablations --config configs/benchmark_edge.yaml \
    --model results/raw/tcn_split01_seed42/model_calibrated.pt \
    --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt \
    --split split01 --role test
```

The inference-time ablations are executed by this script; the ablations that
require retraining (GRU vs TCN, the sequence lengths 8/16/32/64, the
quantisation variants) are produced by the training and quantisation scripts
and collected by `src.evaluation.make_tables`.

## Edge benchmarks

```bash
python -m src.evaluation.benchmark_latency --config configs/benchmark_edge.yaml \
    --model results/raw/tcn_split01_seed42/model_calibrated.pt \
    --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt \
    --warmup 100 --runs 1000

python -m src.evaluation.benchmark_energy --config configs/benchmark_edge.yaml \
    --model results/raw/tcn_split01_seed42/model_calibrated.pt \
    --spatial-checkpoint results/raw/spatial_split01_seed42/model.pt \
    --duration-minutes 30 --compare-frame-skip
```

## Statistics

```bash
python -m src.evaluation.statistics \
    --baseline results/raw/video_eval_baseline.json \
    --candidate results/raw/video_eval_tcn.json \
    --metric detection_delay
```

The unit of analysis is the video. The script performs the paired bootstrap
over the videos, the Wilcoxon signed-rank test and, for binary video-level
outcomes, McNemar's test.
