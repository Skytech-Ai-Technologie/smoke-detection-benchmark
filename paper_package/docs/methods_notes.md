# Methods notes

## 1. System overview

The proposed system is a four-stage cascade that is designed to run on edge
hardware:

```
frame
  -> [1] motion + appearance front end   (Kalman background model, HSV)
  -> [2] spatial smoke classifier        (compact CNN over 50x50 crops)
  -> [3] calibrated temporal evidence    (causal TCN / GRU / accumulator)
  -> [4] interpretable alarm policy      (adaptive threshold and persistence)
```

Stage 1 produces *candidates* (regions that could be smoke), stage 2 scores
each candidate, stage 3 aggregates the scores over time and stage 4 decides
when to raise an alarm.

## 2. Stage 1 - motion and appearance front end

### 2.1 Kalman background model

A per-pixel constant-velocity Kalman filter models the background intensity:

* state `x = [p, v]` with `p` the 2-D position and `v` the velocity,
* observation `z` = the grey intensity of the pixel,
* transition `F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]`,
* process noise `Q = q [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]`,
* measurement noise `R = r I_2`.

A pixel is foreground when the squared, variance-normalised deviation
`dx^2/P_00 + dy^2/P_11` exceeds 9 (a three-sigma criterion). The absolute
deviation between the observation and the *prediction* is kept as the
per-pixel **motion residual**, which is reused as a motion feature.

The covariance is stored as its ten unique entries in flat `(H, W)` float32
planes, so that every operation is a contiguous element-wise pass. This is
about 4.4x faster than a `(H, W, 4, 4)` layout and is mathematically
identical (40 ms per 640x480 frame on the reference CPU).

Smoke is not a rigid object, so the filter is used as a *motion residual*
detector rather than as an object tracker. The ablation "no Kalman motion
model" removes the gating of the candidates by the foreground mask.

### 2.2 HSV candidate extraction

The candidate mask is the intersection of

* the low-saturation constraint `S <= 70` and `V >= 60` (smoke is grey and
  not black), and
* the Kalman foreground mask,

followed by a 5x5 median filter, one morphological opening and two closings
with a 3x3 elliptical kernel, and connected-component analysis. A component
is a candidate when

* its area is at least `0.48 %` of the frame area (resolution invariant; the
  original absolute value of 1200 px corresponds to 640x480),
* its area is at most 60 % of the frame,
* its bounding box has an aspect ratio between 0.6 and 8.0.

At most 8 candidates are kept per frame, the largest first, and each is
cropped to 50x50 pixels for the spatial classifier.

## 3. Stage 2 - spatial smoke classifier

A small CNN over the 50x50 RGB crops:

| Layer | Output |
|---|---|
| Conv 3x3 (16) + BatchNorm + LeakyReLU + MaxPool | 16 x 25 x 25 |
| Conv 3x3 (32) + BatchNorm + LeakyReLU + MaxPool | 32 x 12 x 12 |
| Conv 3x3 (64) + BatchNorm + LeakyReLU + MaxPool | 64 x 6 x 6 |
| Flatten, Dropout 0.5, Linear 2304->128, LeakyReLU, Dropout 0.5, Linear 128->1 | 1 logit |

318 865 parameters, 1.2 MB in FP32. The output is a single logit; the
probability is its sigmoid.

### 3.1 Weakly-supervised training (FIRESENSE)

FIRESENSE has no region-level smoke annotation, so the per-crop label is not
available. The classifier is trained with a multiple-instance objective on
the *frame* label:

```
L = mean over frames of
      [ y * BCE(max_i p_i, 1) + (1 - y) * BCE(mean(top-3 p_i), 0) ]
```

* the positive term uses the **maximum** candidate probability, because the
  detector alarms when any candidate looks like smoke;
* the negative term uses the mean of the **three highest** candidate
  probabilities, which is a smooth upper bound on the maximum and is less
  sensitive to a single outlier crop;
* the two frame classes are balanced with a weight `n_neg / n_pos`.

The number of training epochs is selected by **cross-validation inside the
training split** (3 folds, the epoch with the lowest mean validation loss),
so that the official validation split is never used to fit the spatial model
and the test split is never touched. The same cross-fitting produces the
*out-of-fold* spatial logits of the training videos, which are used to train
the temporal model: without them the temporal model would be trained on
optimistically confident frame probabilities.

### 3.2 Region-level training (D-Fire)

With D-Fire the per-crop label exists (the YOLO boxes of the fire and smoke
regions), and the standard per-crop binary cross-entropy with class
weighting is used. This is the `objective: crop` path.

## 4. Stage 3 - calibrated temporal evidence

### 4.1 Frame features

Every frame is described by a 16-dimensional vector:

| # | Feature | Normalisation |
|---|---|---|
| 0 | maximum spatial smoke probability | identity |
| 1 | mean spatial probability over the candidates | identity |
| 2 | candidate area | / frame area |
| 3 | temporal area change | `log(A_t / A_{t-1})`, clipped to +-3, /3 |
| 4 | motion magnitude (mean residual in the candidate) | /255 |
| 5 | bounding-box width | / frame width |
| 6 | bounding-box height | / frame height |
| 7 | aspect ratio | /20 |
| 8, 9, 10 | mean hue, saturation, value of the candidate | /180, /255, /255 |
| 11 | candidate persistence (consecutive frames of the track) | /64 |
| 12 | frame brightness | /255 |
| 13 | blur score (variance of the Laplacian) | `log1p(b)/10` |
| 14 | number of candidates | /8 |
| 15 | foreground ratio | identity |

The candidate track is a simple IoU tracker (`CandidateTracker`). The
features are standardised with statistics fitted on the **training** split
only.

### 4.2 Causal TCN (the proposed temporal core)

Three dilated causal convolution layers with 16 channels each, kernel size 3
and dilations 1, 2, 4, dropout 0.1, followed by a three-headed output layer
(current probability, probability within the next 8 frames, log-variance).
2 531 parameters. Every convolution is left-padded with its own dilation
factor, so the prediction for time `t` only depends on frames `<= t`
(`tests/test_model_shapes.py::test_tcn_is_causal` verifies this numerically).

### 4.3 GRU

A single-layer GRU with 32 hidden units over the same 16-dimensional
features (4 931 parameters), used as the reference learned temporal model.

### 4.4 Probabilistic evidence accumulator

A non-neural model:

```
l_t = log(p_t / (1 - p_t))
E_t = lambda * E_{t-1} + r_t * l_t
```

with `lambda = 0.95` (forgetting factor) and `r_t` the reliability of the
current observation. The reliability is a logistic function of six
interpretable quality signals (motion consistency, candidate persistence,
blur, brightness, candidate size, model uncertainty); its coefficients are
fitted on the training split. The accumulator is a hysteretic two-state
machine with an alarm threshold of 8.0 and a cancel threshold of -4.0, both
selected on the validation split.

### 4.5 Fixed temporal rules (the mandatory reference)

Mean, median, majority vote, maximum and a fixed persistence rule over a
3-second window of frame probabilities. A learned temporal model is only
interesting if it beats these, and the paper reports them side by side.

### 4.6 Calibration

Four methods are compared on the validation split: none, temperature
scaling, Platt scaling and isotonic regression. The calibration is applied to
the logit of the temporal model and never fitted on the test split.

**Observed result.** On this benchmark, temperature scaling is *not* the best
method: the uncalibrated model has an expected calibration error of 0.18 and
a maximum calibration error of 0.92, temperature scaling does not improve the
ECE (0.22), while Platt scaling reaches an ECE of 0.0015 and isotonic
regression an ECE below 1e-16, with a Brier score of 0.088 instead of 0.174.
The reason is that the error of the model is not a pure temperature
distortion: its output distribution is strongly bimodal, and a single
temperature cannot repair the shape of the score distribution.

## 5. Stage 4 - alarm policy

The policy is a hysteretic state machine with three states (`idle`,
`candidate`, `alarm`):

* the frame probability must exceed the threshold for `k` consecutive frames
  before an alarm is raised, where `k` is the required persistence in
  frames (the persistence is expressed in **seconds** and converted with the
  frame rate of the video),
* an alarm is cleared when the probability falls below
  `hysteresis_ratio x threshold`,
* after an alarm is cleared, a refractory period suppresses new alarms.

### 5.1 Fixed mode

The threshold and the persistence are constants: this is the original rule of
the reference system (`confidence_threshold = 0.70`,
`persistence_seconds = 1.50`).

### 5.2 Adaptive mode

The threshold and the required persistence are modulated by the uncertainty
of the model and by the consistency of the motion evidence:

```
threshold_t = clip(threshold + w_u * uncertainty_t - w_m * motion_reliability_t,
                    threshold_min, threshold_max)
required_t  = clip(persistence * (1 + w_u * uncertainty_t), 0, persistence_max)
```

A high uncertainty therefore raises both the bar and the amount of evidence
that has to accumulate, while a consistent motion signal lowers the bar.
Every quantity is logged with each decision.

### 5.3 Adaptive frame skipping

When the pipeline is in a low-risk state (no candidate in the frame and a
temporal probability below `frame_skip_confidence`), the next
`frame_skip_interval` frames are skipped: the front end and the CNN are not
executed and the previous features are reused (zero-order hold). The
temporal model and the alarm policy still run on every frame, so the
temporal resolution of the *evidence* is preserved while the expensive
spatial stage is executed at a lower rate.

## 6. Interpretation of the decision log

Every frame produces one record:

```json
{
  "video_id": "firesense_testpos10_826",
  "frame_index": 420,
  "timestamp": 14.01,
  "frame_probability": 0.82,
  "calibrated_probability": 0.74,
  "temporal_probability": 0.91,
  "temporal_evidence": 5.93,
  "uncertainty": 0.11,
  "motion_reliability": 0.84,
  "alarm_state": "candidate",
  "decision": "continue_observing",
  "threshold": 0.63,
  "n_candidates": 2,
  "skipped": false
}
```

This log is the artefact that makes an alarm auditable: the probability that
produced it, how confident the model was, how consistent the motion evidence
was, which threshold was in force and which decision the policy took.
