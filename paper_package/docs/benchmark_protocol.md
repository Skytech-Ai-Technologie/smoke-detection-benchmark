# Benchmark protocol

This document defines the protocol of the benchmark **before** the test
splits are evaluated. It is the reference for every number reported in the
paper.

## 1. Data roles

| Role | Used for | Never used for |
|---|---|---|
| **Training** | fitting the model parameters (spatial CNN, temporal model, reliability model, feature standardiser) | reporting |
| **Validation** | hyper-parameters, calibration, the alarm operating point, the number of training epochs (through cross-validation *inside* the training split) | reporting |
| **Test** | the final reporting, once per model | any fitting |
| **External** | cross-dataset generalisation | any fitting |

The test split is touched exactly once per model and per split. The test
manifests are locked read-only after the splits are created and their hashes
are recorded in `data/manifests/splits_locked.json`.

## 2. Splits

* All splits are made at the **video** level. Frames of a video never appear
  in two roles, and files that belong to the same recording session stay
  together. The property is verified by `tests/test_no_video_leakage.py`.
* Five predefined splits (`split01` ... `split05`) are generated from the same
  manifest with the same seed, so that the five splits are reproducible and
  *not* nested.
* Every split contains 22 videos: 12 training (7 positive, 5 negative),
  3 validation (2 positive, 1 negative) and 7 test (4 positive, 3 negative).
* A leave-one-video-out registry is provided as a sensitivity analysis.
* The results are reported as the **mean and the standard deviation over the
  five splits** and per video, never as a single number.

## 3. Metric definitions

### 3.1 Frame-level metrics

For a set of frames with binary ground truth `y` and a continuous score `s`
(the calibrated probability of the detector):

* precision, recall, F1 at the decision threshold (0.5 by default),
* balanced accuracy, Matthews correlation coefficient,
* AUROC, area under the precision-recall curve (average precision),
* confusion matrix at the decision threshold.

The frame label of a video is 1 from the annotated smoke onset onwards and 0
before it. **Caveat**: because the onsets of 12 of the 13 positive videos are
at frame 0 (see the dataset card), the frame-level metrics are close to the
video-level metrics and the frame count must never be used as the effective
sample size.

### 3.2 Alarm events

An **alarm event** is a maximal run of consecutive frames in which the alarm
state is `alarm`. Two events that are separated by less than the refractory
period (2 s by default) are merged into a single event. A single
false-positive *frame* is therefore not an alarm.

### 3.3 Video-level and safety metrics

| Metric | Definition |
|---|---|
| `video_recall` | fraction of positive videos with an alarm raised at or after the onset |
| `video_specificity` | fraction of negative videos without any alarm event |
| `false_alarm_events` | number of alarm events in the negative videos |
| `false_alarms_per_hour` | `false_alarm_events / total_negative_video_hours` |
| `alarms_per_negative_video` | `false_alarm_events / n_negative_videos` |
| `detection_delay` | `first_alarm_time - onset_time` of the first alarm at or after the onset |
| `time_to_first_alarm` | `first_alarm_time` measured from the start of the video |
| `recall_at_{0.5,1.0,2.0}s` | fraction of positive videos detected within that delay |
| `alarm_duration` | total time in the alarm state |
| `alarm_reset_count` | number of alarm events minus one |

A positive video without an alarm is **censored**: it counts against the
video recall but not in the delay distribution. The number of censored videos
is always reported together with the delay.

`detection_delay` and `time_to_first_alarm` coincide when the onset is at
frame 0, which is the case for 12 of the 13 positive videos.

### 3.4 Calibration metrics

Expected calibration error (ECE, 10 equal-width bins, sample weighted),
maximum calibration error (MCE) and the Brier score, reported

* overall,
* for positive and negative frames separately,
* for positive and negative videos separately,
* for bright and dark scenes (split at the median brightness),
* for low-motion and high-motion scenes (split at the median motion).

### 3.5 Edge metrics

Mean, p50, p90, p95 and p99 latency, jitter (the mean absolute deviation of
the frame time around its mean), the number of dropped frames (frames whose
processing time exceeds the frame budget `1/fps`), the peak RSS, the peak
CUDA memory, the CPU time per frame, the estimated energy per frame and the
device temperature.

The latency is measured **per stage**: decoding, Kalman, HSV and candidate
extraction, spatial CNN, feature assembly, temporal model, alarm policy and
logging. The reported system frame rate is the throughput of the **complete**
pipeline; the CNN inference speed alone is never reported as the system
performance.

The energy figure is an estimate: the CPU time per frame is measured and
converted to joules with the configured effective power of the compute
device, which is always reported alongside the number. When a GPU is used,
the package power is sampled with NVML and integrated over the wall-clock
time.

## 4. Model selection and operating points

### 4.1 The operating point is selected on the validation split

For every model, the confidence threshold and the persistence duration are
swept on the **validation** split and the operating point is selected with
the pre-registered rule

> maximise the video recall subject to `false_alarms_per_hour <= 1.0`;
> if no point satisfies the budget, maximise
> `video_recall - penalty * false_alarms_per_hour` with `penalty = 1.0`.

The selected point is then applied to the test split **unchanged**. The
selection writes the full sweep curve to `results/raw/thresholds_*.json`.

### 4.2 The matched-recall comparison

Because a single operating point is brittle on a data set with 22 videos, the
primary comparison of hypothesis H1 is additionally made at **matched
recall**: for every model the false-alarm rate is swept over the evaluation
videos and the models are compared by interpolating every curve at a common
set of recall levels. The sweep is a *reporting* tool; the operating point of
the headline numbers always comes from the validation split.

### 4.3 The epoch count of the spatial model

The number of training epochs of the spatial classifier is selected by
**cross-validation inside the training split** (3 folds), never on the
validation split and never on the test split.

## 5. Statistics

The unit of analysis is the **video**. Concretely:

* the **paired bootstrap** over the videos resamples the per-video metric
  differences (10 000 resamples, 95 % interval, two-sided bootstrap
  p-value);
* the **Wilcoxon signed-rank test** is applied to the paired per-video
  metric differences;
* **McNemar's test** is used for paired binary video-level outcomes (e.g.
  "was the video detected?").

`false_alarms_per_hour` is a corpus-level quantity; for the paired tests it
is decomposed into the per-video number of false alarm events, so that the
resampling unit stays the video.

Frame counts are **never** used as the effective sample size, because the
frames of a video are temporally correlated.

## 6. Ablations

Every ablation uses the same video split and reports the mean and the
standard deviation over the splits and over the seeds:

| # | Ablation | How |
|---|---|---|
| 1 | no Kalman motion model | the foreground mask does not gate the candidates |
| 2 | no HSV filtering | the low-saturation constraint is dropped |
| 3 | CNN only | no temporal model, fixed threshold |
| 4 | CNN plus fixed persistence | the persistence rule of the reference system |
| 5 | GRU versus TCN | two trained temporal models |
| 6 | TCN without handcrafted motion features | the motion, area-change and persistence features are zeroed |
| 7 | TCN without image-quality features | the brightness and blur features are zeroed |
| 8 | no calibration | the raw sigmoid output |
| 9 | temperature scaling versus isotonic regression | two calibrators |
| 10 | fixed threshold versus adaptive threshold | two policy modes |
| 11 | no frame skipping versus adaptive frame skipping | two policies |
| 12 | FP32 versus FP16 versus INT8 | three exported variants |
| 13 | sequence length 8, 16, 32, 64 | four trained models |
| 14 | different false-alarm penalties | the selection penalty of the operating point |
| 15 | different alarm hysteresis values | 0.3, 0.6, 0.9 |

## 7. Pre-registered success margins

The following margins were fixed before the test splits were evaluated. They
are *engineering* targets, not scientific claims.

| Criterion | Margin |
|---|---|
| `minimum_false_alarm_reduction` | 0.20 (20 % fewer false alarms per hour at matched recall) |
| `maximum_allowed_recall_drop` | 0.02 |
| `maximum_allowed_detection_delay_increase_seconds` | 0.25 |
| `maximum_allowed_energy_increase_percent` | 10 |

A hypothesis is considered supported when the criterion is met **and** the
paired statistical test over the videos does not contradict it.

## 8. Definition of success

The study is successful if it demonstrates at least one of the following
under strict video-level evaluation:

* lower false alarms per hour at matched recall,
* lower detection delay at matched false-alarm rate,
* better calibration without a significant loss of recall,
* lower energy per frame at matched temporal performance,
* better external-video generalisation than fixed persistence and simple
  voting.

A negative result is a valid outcome of the study, provided that the
benchmark is reproducible and the analysis explains *why*.
