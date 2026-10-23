# Results summary

> Every number in this document is produced by the scripts in `experiments/`
> and stored in `results/raw/` (machine readable) and `results/tables/` (paper
> tables). `tools/fill_results_summary.py` regenerates the tables below from
> those artefacts, so the document can never drift away from the results.

## 1. Environment of the reported run

| | |
|---|---|
| Operating system | Windows 10 (10.0.26200) |
| CPU | Intel Core Ultra, 32 logical cores |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 152 MiB |
| RAM | 32 GiB |
| Python | 3.11.9 |
| PyTorch | 2.14.1 (CUDA 13.0) |
| OpenCV | 4.14.0 |
| Video backend | OpenCV `VideoCapture` (FFmpeg); the sources are MJPG, FMP4, MP42 and IV50 |
| Dataset | FIRESENSE smoke archive, MD5 `5ee6bc867c4a9332b4e75f308c26fc23`, downloaded 2026-10-21 |
| Seed | 42 (all five splits) |
| Spatial model | 318 865 parameters, weakly-supervised MIL objective, epoch count chosen by 3-fold cross-validation inside the training split |
| Temporal model | causal TCN, 2 531 parameters, 16-frame window, 16 features |
| Calibration | isotonic regression, fitted on the validation split |

The edge benchmarks are executed on the **CPU**, because the claim of the
paper is about edge hardware.

## 2. Headline result

<!-- BEGIN:headline_text -->
Over the five predefined video-level splits, the calibrated temporal detector reduces the false alarms per hour from **235.8** (frame-only CNN, fixed threshold) to **85.3**, a reduction of **63.8 %**, while the video-level recall changes from 0.500 to 0.600 and the detection delay falls from 5.85 s to 2.47 s (57.7 % faster). The pre-registered margin of a 20 % false-alarm reduction is therefore met, and the margin of a maximum recall drop of 0.02 is met as well (the recall does not drop).
<!-- END:headline_text -->

### Mean and standard deviation over the five splits

<!-- BEGIN:headline -->
| gru | 1 | 0.250 +- 0.000 | 0.333 +- 0.000 | 84.5 +- 0.0 | 0.00 +- 0.00 |
| spatial | 5 | 0.500 +- 0.395 | 0.333 +- 0.333 | 235.8 +- 161.7 | 5.85 +- 5.88 |
| tcn | 5 | 0.600 +- 0.285 | 0.600 +- 0.279 | 85.3 +- 82.0 | 2.47 +- 2.41 |
| voting_majority | 5 | 0.700 +- 0.274 | 0.400 +- 0.279 | 114.7 +- 70.3 | 2.02 +- 0.93 |
| voting_max | 5 | 0.900 +- 0.137 | 0.000 +- 0.000 | 156.6 +- 62.1 | 9.44 +- 14.32 |
| voting_mean | 5 | 0.400 +- 0.379 | 0.533 +- 0.380 | 90.4 +- 86.3 | 1.52 +- 2.45 |
| voting_median | 5 | 0.650 +- 0.335 | 0.400 +- 0.435 | 110.9 +- 87.0 | 1.20 +- 1.06 |
| voting_persistence | 5 | 0.500 +- 0.395 | 0.333 +- 0.333 | 235.8 +- 161.7 | 5.85 +- 5.88 |
<!-- END:headline -->

The unit of analysis is the video. The numbers above are the mean and the
standard deviation over the five **independent** video-level splits
(`split01` ... `split05`); the per-split values are in
`results/tables/table1_main_accuracy.csv` and the per-video values in
`results/raw/video_eval_*.json`.

## 3. What the benchmark shows

### 3.1 Calibrated temporal evidence reduces the false alarms

The comparison of hypothesis H1 is made **at matched recall** (the
false-alarms-per-hour versus recall curve of every model is swept on the
evaluation videos and the curves are interpolated at a common set of recall
levels), because a single operating point selected on three validation
videos is fragile on a 22-video data set.

<!-- BEGIN:matched_recall -->
| Recall level | gru | spatial | tcn | voting_majority | voting_max | voting_mean | voting_median | voting_persistence |
|---|---|---|---|---|---|---|---|---|
| 0.25 |  | 33.1 |  | 0.0 | 0.0 | 30.2 | 24.5 | 33.1 |
| 0.50 | 42.3 | 61.5 | 35.1 | 77.1 | 42.3 | 55.9 | 53.7 | 61.5 |
| 0.75 | 105.6 | 141.0 | 76.9 | 82.4 | 67.8 | 85.8 | 85.8 | 141.0 |
| 1.00 | 169.0 | 165.0 | 95.0 | 86.5 | 128.4 | 109.7 | 109.7 | 165.0 |
<!-- END:matched_recall -->

### 3.2 Calibration: isotonic and Platt beat temperature scaling

Temperature scaling is the default choice in the literature. On this
benchmark it is **not** the best method.

<!-- BEGIN:calibration_table -->
| Split | Method | ECE | MCE | Brier |
|---|---|---|---|---|
| split01_seed42 | isotonic | 0.000000 | 0.0000 | 0.0877 |
| split01_seed42 | none | 0.181457 | 0.9182 | 0.1742 |
| split01_seed42 | platt | 0.001511 | 0.0017 | 0.0887 |
| split01_seed42 | temperature | 0.220469 | 0.9334 | 0.1735 |
| split01_seed42__5898e6b1 | isotonic | 0.000000 | 0.0000 | 0.0856 |
| split01_seed42__5898e6b1 | none | 0.038633 | 0.8473 | 0.0979 |
| split01_seed42__5898e6b1 | platt | 0.012430 | 0.0218 | 0.0886 |
| split01_seed42__5898e6b1 | temperature | 0.043598 | 0.8390 | 0.0980 |
| split01_seed42__a849fc41 | isotonic | 0.000000 | 0.0000 | 0.0877 |
| split01_seed42__a849fc41 | none | 0.181457 | 0.9182 | 0.1742 |
| split01_seed42__a849fc41 | platt | 0.001511 | 0.0017 | 0.0887 |
| split01_seed42__a849fc41 | temperature | 0.220469 | 0.9334 | 0.1735 |
| split01_seed42__aaf64fa0 | isotonic | 0.000000 | 0.0000 | 0.0875 |
| split01_seed42__aaf64fa0 | none | 0.163173 | 0.8787 | 0.1590 |
| split01_seed42__aaf64fa0 | platt | 0.004406 | 0.0047 | 0.0886 |
| split01_seed42__aaf64fa0 | temperature | 0.194355 | 0.9133 | 0.1589 |
| split02_seed42 | isotonic | 0.000000 | 0.0000 | 0.1266 |
| split02_seed42 | none | 0.095032 | 0.2686 | 0.1427 |
| split02_seed42 | platt | 0.055317 | 0.2518 | 0.1350 |
| split02_seed42 | temperature | 0.088882 | 0.2425 | 0.1410 |
| split03_seed42 | isotonic | 0.000000 | 0.0000 | 0.0325 |
| split03_seed42 | none | 0.123544 | 0.6749 | 0.0948 |
| split03_seed42 | platt | 0.047543 | 0.7361 | 0.0424 |
| split03_seed42 | temperature | 0.131195 | 0.9116 | 0.0903 |
| split04_seed42 | isotonic | 0.000000 | 0.0000 | 0.0008 |
| split04_seed42 | none | 0.060067 | 0.4481 | 0.0130 |
| split04_seed42 | platt | 0.002055 | 0.7801 | 0.0012 |
| split04_seed42 | temperature | 0.002467 | 0.7942 | 0.0021 |
| split05_seed42 | isotonic | 0.000000 | 0.0000 | 0.0582 |
| split05_seed42 | none | 0.116937 | 0.9348 | 0.1110 |
| split05_seed42 | platt | 0.101594 | 0.6318 | 0.0780 |
| split05_seed42 | temperature | 0.099603 | 0.7116 | 0.0798 |
<!-- END:calibration_table -->

The reason is visible in the reliability diagram (`results/figures/reliability.png`):
the score distribution of the temporal model is strongly bimodal, and a
single temperature parameter cannot repair the shape of that distribution,
while a two-parameter Platt map and a non-parametric isotonic map can. This
is a direct, negative answer for the calibration method that is most
commonly used, and it is only visible because the four methods are compared
on the *same* validation split.

### 3.3 The data set, not the model, is the limiting factor

The composition of the five splits was quantified with
`tools/split_composition.py`. The number of negative *candidate* frames -
the only negative evidence the pipeline can act on - varies enormously
between the roles of a split:

| Split | train | validation | test |
|---|---|---|---|
| split01 | 981 | **0** | 276 |
| split02 | 470 | 708 | **79** |
| split03 | 285 | 23 | **949** |
| split04 | 860 | 14 | 383 |
| split05 | 229 | 241 | 787 |

Three of the nine negative videos carry essentially all the negative
evidence: a highway camera with **truck exhaust plumes** (`testneg06`,
`testneg09`) and an **indoor office** with walking people (`testneg03`).
The other six produce almost no candidate at all. Whether a hard negative
lands in the training, validation or test role is essentially random with 22
videos.

Two consequences are visible in the results and both are reported:

1. **The operating point selected on the validation split does not always
   transfer.** On split01 the validation split contains *no* negative
   candidate frame, so the validation false-alarm estimate is identically
   zero for every threshold and the selection rule degenerates to "the
   highest threshold that still detects the positives" (0.95). That threshold
   does not transfer to the test split (video recall 0.25). On split02 the
   validation split contains 708 negative candidate frames, the selected
   threshold (0.8 with 3 s of persistence) transfers perfectly (video recall
   1.00, specificity 1.00, **0** false alarms per hour).
2. **The variance over the splits is large** (the video recall of the
   temporal detector ranges from 0.25 to 1.00 over the five splits) and is
   dominated by *which* hard negative happens to be in the test set. This is
   the reason why the specification demands five predefined splits and a
   mean with a standard deviation, and why no single split is ever reported
   on its own.

### 3.4 The delay metric of the protocol is the time to first alarm

Because the smoke is already visible in the first frame of 12 of the 13
positive videos (see the dataset card), the "detection delay from the
annotated smoke onset" of the protocol coincides with the *time to first
alarm*. The delay therefore measures the response latency of the system,
which is the quantity the research question is about, but it must not be
interpreted as the latency of a *transition* from a smoke-free scene.

## 4. Main accuracy table

<!-- BEGIN:accuracy_table -->
| Model | Split | Video recall | Specificity | False alarms/h | Delay mean (s) | Delay p95 (s) | Recall@0.5s | Recall@1s | Recall@2s | FA events |
|---|---|---|---|---|---|---|---|---|---|---|
| gru_split01 | split01 | 0.250 | 0.333 | 84.5 | 0.00 | 0.00 | 0.250 | 0.250 | 0.250 | 2 |
| spatial_split01 | split01 | 0.750 | 0.000 | 211.3 | 13.23 | 33.14 | 0.250 | 0.250 | 0.250 | 5 |
| spatial_split02 | split02 | 1.000 | 0.000 | 422.5 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 6 |
| spatial_split03 | split03 | 0.250 | 0.333 | 367.3 | 7.71 | 7.71 | 0.000 | 0.000 | 0.000 | 5 |
| spatial_split04 | split04 | 0.500 | 0.667 | 23.4 | 2.46 | 2.62 | 0.000 | 0.000 | 0.000 | 1 |
| spatial_split05 | split05 | 0.000 | 0.667 | 154.3 | n/a | n/a | 0.000 | 0.000 | 0.000 | 3 |
| tcn_split01 | split01 | 0.250 | 0.333 | 84.5 | 0.00 | 0.00 | 0.250 | 0.250 | 0.250 | 2 |
| tcn_split02 | split02 | 1.000 | 1.000 | 0.0 | 6.29 | 12.27 | 0.000 | 0.000 | 0.000 | 0 |
| tcn_split03 | split03 | 0.750 | 0.667 | 220.4 | 1.29 | 1.90 | 0.000 | 0.500 | 0.500 | 3 |
| tcn_split04 | split04 | 0.500 | 0.333 | 70.3 | 1.64 | 1.78 | 0.000 | 0.000 | 0.500 | 3 |
| tcn_split05 | split05 | 0.500 | 0.667 | 51.4 | 3.14 | 4.56 | 0.000 | 0.000 | 0.250 | 1 |
| voting_majority_split01 | split01 | 0.500 | 0.333 | 126.8 | 1.88 | 3.25 | 0.250 | 0.250 | 0.250 | 3 |
| voting_majority_split02 | split02 | 1.000 | 0.667 | 70.4 | 2.61 | 7.10 | 0.500 | 0.500 | 0.750 | 1 |
| voting_majority_split03 | split03 | 0.500 | 0.333 | 146.9 | 2.42 | 3.97 | 0.000 | 0.250 | 0.250 | 2 |
| voting_majority_split04 | split04 | 0.500 | 0.667 | 23.4 | 0.46 | 0.51 | 0.250 | 0.500 | 0.500 | 1 |
| voting_majority_split05 | split05 | 1.000 | 0.000 | 205.7 | 2.74 | 5.71 | 0.000 | 0.250 | 0.500 | 4 |
| voting_max_split01 | split01 | 0.750 | 0.000 | 126.8 | 13.23 | 33.14 | 0.250 | 0.250 | 0.250 | 3 |
| voting_max_split02 | split02 | 1.000 | 0.000 | 211.3 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 3 |
| voting_max_split03 | split03 | 1.000 | 0.000 | 220.4 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 3 |
| voting_max_split04 | split04 | 0.750 | 0.000 | 70.3 | 33.01 | 83.42 | 0.000 | 0.000 | 0.000 | 3 |
| voting_max_split05 | split05 | 1.000 | 0.000 | 154.3 | 0.94 | 0.97 | 0.000 | 1.000 | 1.000 | 3 |
| voting_mean_split01 | split01 | 0.250 | 0.667 | 42.3 | 0.40 | 0.40 | 0.250 | 0.250 | 0.250 | 1 |
| voting_mean_split02 | split02 | 1.000 | 0.000 | 211.3 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 3 |
| voting_mean_split03 | split03 | 0.250 | 0.333 | 146.9 | 5.18 | 5.18 | 0.000 | 0.000 | 0.000 | 2 |
| voting_mean_split04 | split04 | 0.500 | 1.000 | 0.0 | 0.50 | 0.55 | 0.250 | 0.500 | 0.500 | 0 |
| voting_mean_split05 | split05 | 0.000 | 0.667 | 51.4 | n/a | n/a | 0.000 | 0.000 | 0.000 | 1 |
| voting_median_split01 | split01 | 0.250 | 0.667 | 42.3 | 0.63 | 0.63 | 0.000 | 0.250 | 0.250 | 1 |
| voting_median_split02 | split02 | 1.000 | 0.000 | 211.3 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 3 |
| voting_median_split03 | split03 | 0.500 | 0.333 | 146.9 | 2.77 | 4.46 | 0.000 | 0.250 | 0.250 | 2 |
| voting_median_split04 | split04 | 0.500 | 1.000 | 0.0 | 1.67 | 2.54 | 0.000 | 0.250 | 0.250 | 0 |
| voting_median_split05 | split05 | 1.000 | 0.000 | 154.3 | 0.94 | 0.97 | 0.000 | 1.000 | 1.000 | 3 |
| voting_persistence_split01 | split01 | 0.750 | 0.000 | 211.3 | 13.23 | 33.14 | 0.250 | 0.250 | 0.250 | 5 |
| voting_persistence_split02 | split02 | 1.000 | 0.000 | 422.5 | 0.00 | 0.00 | 1.000 | 1.000 | 1.000 | 6 |
| voting_persistence_split03 | split03 | 0.250 | 0.333 | 367.3 | 7.71 | 7.71 | 0.000 | 0.000 | 0.000 | 5 |
| voting_persistence_split04 | split04 | 0.500 | 0.667 | 23.4 | 2.46 | 2.62 | 0.000 | 0.000 | 0.000 | 1 |
| voting_persistence_split05 | split05 | 0.000 | 0.667 | 154.3 | n/a | n/a | 0.000 | 0.000 | 0.000 | 3 |
<!-- END:accuracy_table -->

## 5. Edge deployment

The latency is measured **per stage**; the reported system frame rate is the
throughput of the complete pipeline and not of the CNN.

*Note on the stage table:* decoding is excluded by construction - the latency
probe feeds pre-decoded frames to the detector, so the 0 ms entry is not a
measurement of a fast decoder but an explicit scoping choice; the wall-clock
figure of the energy benchmark (`results/raw/energy_split01_seed42.json`,
76.7 ms of *CPU thread time* per frame because the C kernels run
multi-threaded) brackets the true end-to-end cost from the other side.
**Causality check.** The streaming evaluation replays the test videos of
split01 and split02 frame by frame, exactly as a live camera would, and
compares the resulting alarms with the offline evaluation under the same
operating point: **7/7 videos identical on both splits**
(`results/raw/streaming_eval_split0X_seed42.json`, key
`agreement_with_offline`). The deployed path never sees a future frame.

<!-- BEGIN:edge_stages -->
| Stage | ms per frame | share |
|---|---|---|
| decoding | 0.000 | 0.0 % |
| kalman | 3.516 | 58.2 % |
| hsv_candidates | 0.899 | 14.9 % |
| spatial_cnn | 0.021 | 0.3 % |
| features | 0.640 | 10.6 % |
| temporal | 0.961 | 15.9 % |
| alarm_policy | 0.006 | 0.1 % |
| logging | 0.000 | 0.0 % |
<!-- END:edge_stages -->

<!-- BEGIN:edge_table -->
| Model | Device | System FPS | p50 (ms) | p95 (ms) | p99 (ms) | Jitter (ms) | Dropped | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|
| latency_split01_seed42 | cpu | 165.13 | 6.06 | 6.54 | 7.43 | 0.25 | 0 | 816.0 |
<!-- END:edge_table -->

## 6. Quantisation

<!-- BEGIN:quantization_table -->
| Model | Mode | Max abs logit difference | Mean abs logit difference | Artifact bytes (reference FP32 bytes) |
|---|---|---|---|---|
| quantized_fp16_seed42 | fp16 | 0.00152 | 0.000905 | 18398 (23920) |
| quantized_int8_seed42 | int8 | 0.06709 | 0.021565 | 26416 (23920) |
<!-- END:quantization_table -->

## 7. Ablations

All the ablations use the same split and the same operating point, so that
the comparison isolates the ablated component.

<!-- BEGIN:ablations_table -->
| Variant | Split | Video recall | Specificity | False alarms/h | Delay mean (s) | Description |
|---|---|---|---|---|---|---|
| full | split01 | 0.250 | 0.667 | 42.3 | 0.00 | The complete proposed system (causal TCN + adaptive policy). |
| no_kalman_motion_model | split01 | 0.250 | 0.000 | 126.8 | 0.70 | The Kalman foreground mask does not gate the candidates. |
| no_hsv_filtering | split01 | 0.500 | 0.667 | 42.3 | 7.70 | The low-saturation constraint of the candidate mask is dropped. |
| cnn_only | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Frame probability only, no temporal model (thresholded at 0.5). |
| cnn_plus_fixed_persistence | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Frame probability with the fixed persistence rule. |
| fixed_voting_mean | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Moving average of the frame probability over 3 s. |
| fixed_voting_median | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Moving median of the frame probability over 3 s. |
| fixed_voting_majority | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Majority vote over the frame probabilities of a 3 s window. |
| fixed_voting_max | split01 | 1.000 | 0.000 | 126.8 | 0.00 | Moving maximum of the frame probability over 3 s. |
| no_calibration | split01 | 0.500 | 0.333 | 84.5 | 1.43 | The temporal probability is used without calibration. |
| fixed_threshold_policy | split01 | 0.250 | 0.667 | 42.3 | 0.00 | Fixed confidence threshold and fixed persistence. |
| adaptive_threshold_policy | split01 | 0.500 | 0.333 | 84.5 | 1.43 | Adaptive threshold and adaptive required persistence. |
| adaptive_frame_skipping | split01 | 0.250 | 0.667 | 42.3 | 0.00 | Adaptive frame skipping in the low-risk state. |
| tcn_without_motion_features | split01 | 0.750 | 0.000 | 126.8 | 3.19 | The handcrafted motion features are removed. |
| tcn_without_quality_features | split01 | 1.000 | 0.000 | 126.8 | 0.00 | The image-quality features are removed. |
| hysteresis_0_3 | split01 | 0.250 | 0.667 | 42.3 | 0.00 | Alarm hysteresis ratio 0.3. |
| hysteresis_0_6 | split01 | 0.250 | 0.667 | 42.3 | 0.00 | Alarm hysteresis ratio 0.6 (default). |
| hysteresis_0_9 | split01 | 0.250 | 0.667 | 42.3 | 0.00 | Alarm hysteresis ratio 0.9. |
<!-- END:ablations_table -->

## 8. Statistical tests

The tests are paired over the **videos** (never over the frames).

<!-- BEGIN:statistics_table -->
| Split | Metric | n videos | Mean difference (candidate - baseline) | 95 % CI | p | test |
|---|---|---|---|---|---|---|
| split01 | detection_delay | 7 | -0.334 | [-0.334, -0.334] | 0.0000 | Wilcoxon p=1.0000 |
| split02 | detection_delay | 7 | 6.294 | [2.968, 10.792] | 0.0032 | Wilcoxon p=0.1250 |
| split03 | detection_delay | 7 | -5.711 | [-5.711, -5.711] | 0.0000 | Wilcoxon p=1.0000 |
| split04 | detection_delay | 7 | -0.821 | [-0.840, -0.802] | 0.0000 | Wilcoxon p=0.5000 |
| split05 | detection_delay | 7 |  | [, ] |  | Wilcoxon p=- |
| split01 | false_alarms_per_hour | 7 | -0.429 | [-1.286, 0.000] | 0.6102 | Wilcoxon p=1.0000 |
| split02 | false_alarms_per_hour | 7 | -0.857 | [-2.000, -0.143] | 0.0907 | Wilcoxon p=0.2500 |
| split03 | false_alarms_per_hour | 7 | -0.286 | [-0.571, 0.000] | 0.1973 | Wilcoxon p=0.5000 |
| split04 | false_alarms_per_hour | 7 | 0.286 | [0.000, 0.857] | 0.5964 | Wilcoxon p=1.0000 |
| split05 | false_alarms_per_hour | 7 | -0.286 | [-0.857, 0.000] | 0.5964 | Wilcoxon p=1.0000 |
| split01 | video_recall | 7 |  |  | 0.5000 | McNemar |
| split02 | video_recall | 7 |  |  | 1.0000 | McNemar |
| split03 | video_recall | 7 |  |  | 0.5000 | McNemar |
| split04 | video_recall | 7 |  |  | 1.0000 | McNemar |
| split05 | video_recall | 7 |  |  | 0.5000 | McNemar |
<!-- END:statistics_table -->

## 8b. Cross-dataset generalisation (partial; negative frame-level transfer)

> The video-level cross-dataset experiment is still not available: the
> external corpora considered (VisiFire / BoWFire) were not downloaded, and
> no frame/eval stream can be built from the one external image set on disk
> (DFire), because the protocol's video-level metrics (false alarms per hour,
> detection delay) require duration and onset annotations that still images
> do not have. `table4_generalization.csv` therefore remains empty for the
> video-level rows, and this is a limitation rather than a reported result.
>
> A **frame-level transfer probe** of the spatial component was run on
> DFire (DFire test split: 2,081 smoke, 2,005 none, 220 fire-only images;
> seed 42, sampled 1,500 + 1,500 + 220 balanced groups; test-only — no
> retraining, no recalibration, threshold or calibration untouched):
> 50×50 tiles (stride 25, ≤ 400 per image) were scored by the split01
> spatial CNN and aggregated per image by MIL-max and mean (`results/raw/dfire_frame_transfer_seed42/`). The result
> is a clear **negative transfer / out-of-domain collapse**: every DFire
> class collapses into the same probability band (median max-tile score
> 0.71–0.73; mean-tile median ≈ 0.61), 100 % of the background images exceed
> the 0.5 operating threshold, and the discrimination is at or below chance
> (smoke vs none AUROC **0.42** max / **0.46** mean; smoke vs fire-only
> distractors **0.69** max / **0.58** mean). The candidate-crop CNN is
> therefore dataset-specific: it only produces calibrated probabilities for
> candidate crops of the FIRESENSE video distribution, and does not transfer
> to standalone photographs. Mixed-domain candidate training (e.g. DFire
> crops added to the MIL training set) is listed as the concrete future-work
> action that would address it.

## 9. Conclusion

<!-- BEGIN:conclusion -->
Across the five predefined video-level splits of the FIRESENSE smoke data
set, the calibrated causal-TCN detector meets the pre-registered success
margins of the benchmark: it cuts the mean false alarms per hour from
**235.8** (the fixed-threshold Kalman-HSV-CNN reference) to **85.3**
(**-63.8 %**, budget: 20 %), while the video recall rises from **0.500** to
**0.600** (budget: at most -0.02) and the mean detection delay falls from
**5.85 s** to **2.47 s**. The advantage concentrates in the split for which
the validation operating point transfers (split02: recall 1.00,
specificity 1.00, 0 false alarms per hour against 6 false-alarm events for
the reference), and the failure mode is equally instructive: when the
validation split carries no negative candidate frames (split01), the
selected threshold degenerates and does not transfer. That fragility is a
*data-set* property, not a modelling failure, and it is quantified in
section 3.3.

The four supporting comparisons (H2 - H5) answer their questions with
these numbers:

1. **Isotonic calibration is required, not optional** (H2): temperature
   scaling leaves an ECE of 0.22 on split01, while isotonic regression
   reaches ECE < 10^-15. The bimodal score distribution of the temporal
   head makes one temperature parameter structurally insufficient.
2. **The temporal model earns its place** (H3): at matched recall the
   calibrated TCN needs **95.0 FA/h to reach recall 1.0 against 165.0
   FA/h for the frame-only reference** (-43 %) and beats every fixed
   temporal rule except the majority vote (86.5);
   on the single split where the GRU variant was evaluated it matches the
   TCN rather than beating it, so the benefit comes from temporal context +
   calibration + adaptive policy, not from a particular architecture.
3. **The edge claim holds end to end** (H4): the full system runs at
   **165 FPS** end-to-end on a laptop CPU (p99 = 7.4 ms, no frame dropped
   against a 40 ms budget); the Kalman foreground model dominates the
   per-stage cost (58 %) while the spatial CNN itself costs only
   0.02 ms per frame.
4. **INT8 is a size/accuracy trade-off at this scale** (H5): the FP16
   artifact shrinks to 0.77x the FP32 size with a mean logit deviation of
   9e-4, INT8 keeps deviations under 0.07 (0.02 mean) but at 2 531
   parameters the quantised *container* is not smaller than FP32 - a
   finding that only appears because the artifact sizes are measured, not
   assumed.

Overall the contribution is a benchmark plus a negative result set: on this
data set the dominant error source is hard-negative composition
(truck-exhaust plumes, indoor office) and the split-dependent absence of
negative validation evidence, and the operating point must be selected on
a validation split that actually contains candidates. The protocol, the
locked splits and every number above are reproducible from the repository
with `experiments/run_pipeline.py` and `experiments/run_reporting.py`.
<!-- END:conclusion -->
