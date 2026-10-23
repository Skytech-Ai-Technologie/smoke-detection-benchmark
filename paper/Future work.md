# Script Changes for a More Defensible Paper (No Edge Device Required)

This checklist is based on the current repository structure. It focuses on evaluation validity, reproducibility, and statistical reporting. It does **not** require an embedded board or a new hardware implementation.

## 1. Fix the paired-statistics handling first

**File:** `src/evaluation/statistics.py`

There is a concrete issue in the current `detection_delay` comparison: `_per_video_values()` skips a video if either model missed it, but `compare()` still reports `n_paired_videos=len(shared)` and zips the shortened delay arrays against the unfiltered `shared` IDs. This can overstate the paired sample size and attach delay values to the wrong video IDs.

Change the helper to return the IDs actually retained for each metric, for example `paired_ids`, `values_a`, and `values_b`. For delay comparisons, include only positive videos detected by **both** models, and report that paired count explicitly. Separately compare detection success on **all** positive videos using McNemar’s test, and compare recall by fixed deadlines. A missed video must not disappear from the overall safety comparison merely because its delay is undefined.

For false alarms per hour, the current paired comparison uses raw false-alarm event counts per negative video. Since negative videos have different durations, implement a paired cluster bootstrap that resamples negative-video IDs and, in each resample, recomputes:

`sum(false_alarm_events) / sum(negative_video_duration_hours)`

for both systems. Resample the same video IDs for the baseline and candidate. Keep the corpus-level rate and its exposure denominator in every result record.

**Add tests:** `tests/test_statistics.py` should verify that missed detections do not misalign IDs, that reported `n` matches the values used in the test, and that false-alarm rates use exposure time rather than raw counts alone.

## 2. Make test-set threshold curves descriptive, not the primary selected result

**Files:** `experiments/run_pipeline.py`, `src/evaluation/sweep_curve.py`, `src/training/select_thresholds.py`

The current pipeline runs `sweep_curve` on the `test` role. A curve drawn from the test set is useful as a descriptive operating-characteristic plot, but choosing a threshold or target-recall point from that same curve and then presenting it as a selected test result is optimistic.

For every primary operating-point result:

1. select the threshold or alarm policy on validation data only;
2. freeze it and apply it once to the test videos;
3. store the validation rule, selected threshold, persistence duration, model checkpoint hash, and test result together.

If you retain matched-recall curves computed on test labels, identify them as **descriptive test-set curves**, not as validation-selected operating points or independent proof of superiority. For a stronger primary comparison, choose operating points using validation data, then report test recall, false alarms per hour, and missed videos. Because each current validation set is tiny, report when the requested recall target cannot be estimated reliably rather than silently selecting a test threshold.

**Add a leakage test:** fail if a script that writes the paper’s primary operating-point table reads test labels before the threshold/policy is fixed.

## 3. Run a split-by-seed matrix with immutable outputs

**Files:** add `experiments/run_robust_matrix.py`; use `experiments/run_pipeline.py` as the worker.

`run_pipeline.py` already accepts one `--split` and one `--seed`. Add a small matrix runner that invokes it for all five locked splits and at least three fixed training seeds (for example, 42, 43, and 44). If compute time permits, use five seeds. Give each run a unique output directory and preserve its stdout/stderr, command, commit, config hashes, split-manifest hash, seed, device, start/end time, and exit code in a machine-readable run manifest. Stop or mark failures explicitly; do not overwrite a previous run.

Example shell loop for a first three-seed pass:

```bash
for seed in 42 43 44; do
  for split in split01 split02 split03 split04 split05; do
    python experiments/run_pipeline.py \
      --split "$split" \
      --seed "$seed" \
      --device cpu \
      --out-dir "results/raw/seed_${seed}"
  done
done
```

Use `--device cuda` instead if an ordinary GPU is available; this is still not an edge-device claim. Never reuse a checkpoint across different seeds or splits. Report split-to-split variability separately from seed-to-seed variability. Fifteen runs are **not** fifteen independent test datasets: the inference unit remains the video.

## 4. Make delay reporting explicitly miss-aware

**Files:** `src/evaluation/metrics.py`, `src/evaluation/evaluate_video.py`, `tests/test_metrics.py`

The current code already reports `n_missed_positive` and recall at 0.5, 1.0, and 2.0 seconds. Preserve these. Add recall by a predeclared longer deadline (for example, 5 seconds), and export a per-positive-video row containing onset, video duration, detected/not detected, first alarm time, delay if detected, and remaining observation time if missed.

For each model, report both:

- conditional delay among detected clips, with the number of missed clips next to it; and
- miss-aware recall by deadline and the count of misses.

Do not replace a missed delay with zero or with the clip duration and then take an ordinary median. If you add survival analysis, mark non-detections as right-censored, state the censoring assumptions, and keep it descriptive because the positive sample is very small. The existing paired tests should not treat a missed clip as a missing-at-random delay.

**Tests:** add cases for an alarm before onset, an alarm after onset, no alarm, zero-frame onset, and different video durations. Verify that a miss lowers recall and is counted as censored without becoming a zero-second delay.

## 5. Correct calibration uncertainty for correlated frames

**File:** `src/evaluation/evaluate_calibration.py`

The current script concatenates frame predictions and computes calibration metrics over all frames. Frames in a single video are strongly correlated, and long videos contribute more observations. Keep frame-level ECE/Brier as descriptive metrics if useful, but add per-video Brier scores and confidence intervals obtained by resampling **whole videos**, not individual frames. For reliability plots, retain the video ID for each frame so uncertainty can be clustered by video. Use only validation data to fit calibration; use test data only for held-out reporting.

Do not describe a metric as “video-level calibration” merely because positive and negative videos were used to label frames. Define the aggregation unit in the output and in the paper. Treat ECE as secondary because it depends on binning; include Brier score and the reliability plot, with video counts shown.

## 6. Rerun only a small, predeclared ablation set across the matrix

**File:** `src/evaluation/run_ablations.py`; orchestration in `experiments/run_robust_matrix.py`.

The current ablation CLI runs one split and one seed. The repository’s visible `ablations_split01_seed42.json` is not sufficient to support a statement that the ablations are averaged over all splits and seeds. Either correct the paper/protocol to match the single run or rerun a focused set across the same split-by-seed matrix.

Prioritize these four comparisons:

1. frame-only CNN versus causal TCN;
2. TCN with versus without calibration;
3. TCN with versus without the main handcrafted motion/persistence features;
4. proposed alarm policy versus one simple fixed temporal baseline.

For a variant that changes the score distribution, choose its threshold using validation data for that variant; do not blindly reuse the full model’s threshold. Keep the test set out of ablation selection. Record each variant, seed, split, selected validation policy, checkpoint hash, and per-video outputs. If a component is only changed at inference, state that; do not imply the model was retrained when it was not.

## 7. Add a single source-of-truth validation script

**File:** add `tools/validate_paper_artifacts.py`.

Have it check that:

- every expected model/split/seed has a result file, or an explicit failure record;
- there are no duplicate run IDs or overwritten outputs;
- all result files point to the correct locked split and code commit;
- the primary table is regenerated from raw JSON rather than manually edited;
- manuscript numbers match the generated table within a stated tolerance;
- the paper and README do not call overlapping partitions “independent”; and
- the reported ablation coverage matches the actual run inventory.

Run this validation script in CI along with `pytest -q`. Add unit tests for the specific statistical and censoring cases above before launching the full run matrix.

## 8. Do not expand edge scripts for this paper revision

You do not need to add a new device-specific branch to `benchmark_latency.py` or `benchmark_energy.py`. Keep laptop CPU throughput as a secondary software-runtime measurement, state that decoding is excluded, and describe the energy figure as an estimate if it is derived from configured power. Move embedded latency, power-meter measurements, sustained operation, and camera-to-alarm timing to Future Work. Avoid using “edge-ready,” “low-power,” or “real-time deployment” as empirical conclusions without those measurements.

## Recommended implementation order

1. Fix the paired-delay ID/count bug and duration-normalized false-alarm bootstrap in `statistics.py`.
2. Fix the test-threshold-sweep interpretation and freeze validation-selected policies.
3. Add the miss-aware per-video outputs and tests.
4. Add the split-by-seed runner and immutable run manifest.
5. Add video-clustered calibration uncertainty and the four focused ablations.
6. Regenerate all tables/figures, run the artifact validator, and revise paper claims to match the completed experiments.

This sequence targets the paper’s most consequential validity risks first and does not depend on an embedded device.
