# Model card

## 1. Model overview

| | |
|---|---|
| **Name** | Calibrated temporal smoke detector (causal TCN) |
| **Task** | binary smoke / no-smoke decision on video frames, with a video-level alarm |
| **Input** | an RGB video stream (the benchmark uses 300x240 to 704x576) |
| **Output** | a calibrated per-frame smoke probability, a temporal probability, an uncertainty estimate and an alarm state |
| **Stage 1** | per-pixel Kalman background model + HSV candidate extraction (no parameters) |
| **Stage 2** | spatial CNN over 50x50 candidate crops, 318 865 parameters |
| **Stage 3** | causal TCN over 16 frame features, 2 531 parameters |
| **Stage 4** | hysteretic alarm policy with an adaptive threshold |
| **Total trainable parameters** | 321 396 (0.4 % of the reference baseline is retained) |
| **Precision** | FP32, FP16 and INT8 variants are exported |
| **Licence** | Apache-2.0 (code) |

## 2. Intended use

* Research on smoke detection in video with strict video-level evaluation.
* Edge deployment on platforms with a CPU or a low-power accelerator, where
  the whole pipeline has to run in real time.

**Out of scope / not recommended for**

* open-fire detection in unconstrained outdoor scenes (the benchmark is
  dominated by indoor and short-range outdoor recordings),
* safety-critical decisions without human review,
* any claim of generalisation to a sensor, a camera type or a fire regime
  that is not represented in the benchmark.

## 3. Training data

* **FIRESENSE** (Zenodo 836749, CC BY 4.0): 22 videos, 13 positive and 9
  negative, 17 722 frames. The temporal benchmark.
* **D-Fire** (GitHub, region-level YOLO labels): the intended source of
  spatial supervision; the archive requires manual acquisition, see
  `configs/data_dfire.yaml` and the dataset card.

No private data is used, and no video of the AdViSED/railway dataset is part
of the reproducibility claim.

## 4. Method

See `paper/methods_notes.md` for the full description. In short:

1. a per-pixel Kalman background model and an HSV low-saturation filter
   produce a small number of *candidates* per frame;
2. a compact CNN scores every candidate; on FIRESENSE it is trained with a
   weakly-supervised (multiple-instance) objective because the dataset has no
   region-level annotation;
3. a **causal** dilated TCN aggregates the frame features over a 16-frame
   window and outputs the current probability, the probability within the next
   8 frames and an uncertainty estimate;
4. a hysteretic alarm policy raises an alarm when the calibrated probability
   stays above an adaptive threshold for a required persistence; every
   decision is logged with its evidence.

## 5. Evaluation

The model is evaluated with strict video-level splits (five predefined
splits, never a random frame split) and reports video recall, video
specificity, false alarms per hour, detection delay, calibration error, the
end-to-end latency per stage and the energy per frame. The unit of analysis
is the video.

## 6. Performance

The measured numbers of the reported run are in `paper/results_summary.md`
and `results/tables/`. Two caveats are part of the reported performance:

* the data set contains **22 videos only**; a single video changes the video
  recall by 7.7 points, so all conclusions are reported as a mean and a
  standard deviation over the five splits;
* the operating point selected on the validation split of a split does not
  always transfer to its test split, because the validation split contains
  one negative video only. The paper therefore also reports the
  **false alarms per hour versus recall** curve and compares the models at
  **matched recall**.

## 7. Robustness and failure modes

| Failure mode | Behaviour | Mitigation |
|---|---|---|
| Vehicle exhaust, steam, fog, dust | low-saturation moving region that resembles smoke | the hard negatives `testneg03`, `testneg06`, `testneg09`; the temporal consistency features |
| Fire that does not produce a visible plume (flame only) | the HSV candidate mask stays empty | out of scope: this is a *smoke* benchmark |
| Very low frame rate (4.7 fps in the benchmark) | the persistence in seconds covers very few frames | all duration parameters are expressed in seconds |
| Small or distant plume | the candidate area falls below the threshold | the area threshold is a fraction of the frame area, which keeps the filter resolution invariant |
| Night / very dark scenes | the value constraint `V >= 60` removes the candidate | the benchmark contains one dark positive video (`testpos05`); the calibration is reported separately for bright and dark scenes |
| Camera motion | the Kalman model reports the whole frame as foreground | the candidate is additionally constrained by the HSV mask and the area limit |

## 8. Ethical considerations and intended audience

The system is a research artefact. Smoke detection is used in wildfire early
warning and in the protection of cultural heritage, and a false negative can
have serious consequences; the system must not be deployed without a human
in the loop and without a site-specific validation. The false-alarm rate
reported here is measured on 22 videos and does not transfer to an
operational setting without re-validation.

## 9. Reproducibility

* `experiments/run_pipeline.py` reproduces one split end to end.
* `experiments/registry.csv` records the git commit, the manifest hash, the
  config hash, the seed, the device, the precision, the command line and the
  checkpoint hash of every run.
* The dataset archives are verified by MD5 and SHA-256.
* The unit tests verify the absence of video leakage, the causality of the
  temporal model and the metric definitions.
