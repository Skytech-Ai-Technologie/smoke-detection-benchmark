# Dataset card

## 1. Overview

| | |
|---|---|
| **Name** | FIRESENSE (smoke subset) |
| **Primary role** | temporal benchmark (primary) |
| **Source** | Zenodo record 836749, <https://zenodo.org/records/836749> |
| **DOI** | 10.5281/zenodo.836749 |
| **Licence** | CC BY 4.0 |
| **Files used** | `smoke_videos.1407.zip` (197 589 846 bytes) |
| **Verified** | MD5 `5ee6bc867c4a9332b4e75f308c26fc23`, SHA-256 `876035a132f81119504e9fa53634d56215dd84b29c986d373bff4ee638b37eea` |
| **Download date** | 2026-10-21 |
| **Content** | 13 positive and 9 negative videos, 17 722 frames, 833.8 s (0.231 h) |

The archive was downloaded with `python -m src.data.download --config
configs/data_firesense.yaml`, which verifies the size, the MD5 and the
SHA-256 and refuses to replace an existing file whose hash differs.

## 2. Video inventory

| Video | Label | Frames | FPS | Resolution | Duration (s) | Scene |
|---|---|---|---|---|---|---|
| testneg01.807 | negative | 251 | 25.0 | 352x288 | 10.0 | outdoor, low motion |
| testneg02.808 | negative | 1128 | 25.0 | 320x240 | 45.1 | outdoor, low motion |
| testneg03.809 | negative | 182 | 30.0 | 640x480 | 6.1 | **indoor office, walking people (hard negative)** |
| testneg04.810 | negative | 2840 | 25.0 | 704x576 | 113.6 | outdoor, low motion |
| testneg05.811 | negative | 251 | 25.0 | 352x288 | 10.0 | static scene, no candidate at all |
| testneg06.812 | negative | 251 | 25.0 | 352x288 | 10.0 | **highway, truck exhaust plumes (hard negative)** |
| testneg07.813 | negative | 776 | 25.0 | 352x288 | 31.0 | outdoor |
| testneg08.814 | negative | 751 | 25.0 | 352x288 | 30.0 | outdoor, camera motion |
| testneg09.815 | negative | 723 | 30.0 | 640x480 | 24.1 | **highway, dense traffic (hard negative)** |
| testpos01.817 | smoke | 407 | 24.0 | 640x360 | 17.0 | lawn, small plume |
| testpos02.818 | smoke | 1728 | 30.0 | 640x424 | 57.7 | building, heavy plume + flame |
| testpos03.819 | smoke | 322 | 15.0 | 320x240 | 21.5 | hillside |
| testpos04.820 | smoke | 1964 | 25.0 | 640x360 | 78.6 | gravel path, two plumes |
| testpos05.821 | smoke | 514 | 4.7 | 320x240 | 109.6 | dark indoor, plume |
| testpos06.822 | smoke | 201 | 10.0 | 320x240 | 20.1 | person igniting a smoke pot |
| testpos07.823 | smoke | 701 | 10.0 | 320x240 | 70.1 | person over a smoke pot |
| testpos08.824 | smoke | 363 | 15.0 | 320x240 | 24.2 | shed, smoke pot + flame |
| testpos09.825 | smoke | 828 | 30.0 | 640x360 | 27.6 | lawn, smoke pot |
| testpos10.826 | smoke | 1469 | 30.0 | 480x360 | 49.0 | large plume |
| testpos11.827 | smoke | 677 | 25.0 | 320x240 | 27.1 | barrel, plume |
| testpos12.828 | smoke | 362 | 30.0 | 480x270 | 12.1 | dense smoke, person |
| testpos13.829 | smoke | 463 | 30.0 | 300x240 | 15.5 | large plume |

**Heterogeneity.** The frame rate ranges from 4.7 to 30 fps and the
resolution from 300x240 to 704x576. Every duration-dependent parameter is
therefore expressed in **seconds** and converted with the frame rate of the
individual video, and every area threshold is expressed as a **fraction of the
frame area** (`min_area_ratio = 0.0048`, i.e. the 1200 px of the original
640x480 configuration), so that the candidate filter is resolution invariant.

## 3. Labels and the missing annotation

The public release contains the raw videos and **no annotation file**. The
following facts were established while preparing the benchmark and they shape
every number in the paper.

### 3.1 The onsets were annotated manually

Protocol (implemented in `tools/make_contact_sheet.py`, sheets in
`data/annotations/sheets/` and `data/annotations/sheets_fine/`):

> The onset of a positive video is the **first frame at which smoke is
> unambiguously visible to a human annotator**. A 4x4 contact sheet of the
> whole video and a finer 4x4 contact sheet of the first 120 frames were
> inspected for every positive video.

The result is in `data/annotations/onsets.csv` and in the `onset_frame`
column of `data/manifests/firesense_videos.csv`.

| Video | Onset (frame) | Time (s) | Official burned-in contour | Note |
|---|---|---|---|---|
| testpos01 | 15 | 0.63 | – | faint smudge from f0, distinct puff from f15 |
| testpos02 | 0 | 0.00 | – | heavy plume and flame in the first frame |
| testpos03 | 0 | 0.00 | – | small plume at the bottom of the frame |
| testpos04 | 0 | 0.00 | – | two clear plumes in the first frame |
| testpos05 | 0 | 0.00 | 36 | plume visible from f0 |
| testpos06 | 0 | 0.00 | – | full smoke column from the pot in the first frame |
| testpos07 | 0 | 0.00 | – | small wisp above the pot in the first frame |
| testpos08 | 0 | 0.00 | – | thin plume and flame at the right of the shed |
| testpos09 | 0 | 0.00 | – | clear plume above the smoke pot |
| testpos10 | 0 | 0.00 | – | large plume in the first frame |
| testpos11 | 0 | 0.00 | 150 | small puff above the barrel |
| testpos12 | 0 | 0.00 | 56 | dense smoke in the first frame |
| testpos13 | 0 | 0.00 | – | large plume in the first frame |

The *official burned-in contour* is the cyan smoke-region outline that the
FIRESENSE recording tool drew into a subset of the videos. It is detected
automatically by `tools/detect_annotations.py` and recorded as a cross-check
only. It is systematically **late** (it appears only once the region is large
enough to be drawn), which is why it is not used for the primary metric.

### 3.2 Consequence: the onset coincides with the start of the clip

In 12 of the 13 positive videos the smoke is already visible in the very
first frame. The positive videos therefore contain **almost no pre-onset
frames**. Three consequences are reported throughout the paper:

1. **The "detection delay from the annotated onset" is the "time to first
   alarm".** The metric is still computed and reported under the name of the
   protocol, but it must be read as the response latency of the system: the
   time from the start of the clip until the first alarm. The paper states
   this explicitly wherever the delay is discussed.
2. **The frame-level metrics are close to the video-level metrics**, because
   the positive frames of the benchmark are almost exactly the frames of the
   positive videos.
3. **The pre-onset negative frames are scarce.** A classifier trained on
   them would be trained on almost no background of the positive scenes; the
   negative videos carry that information instead.

### 3.3 Consequence: the spatial classifier needs a weakly-supervised objective

Because there is no region-level annotation, labelling every candidate of a
smoke frame as "smoke" would teach the network that every low-saturation
region is smoke (including the sky). The spatial classifier is therefore
trained with a **multiple-instance objective** on the *frame* label:

* a **positive** frame must contain at least one smoke-like candidate — the
  loss is applied to the maximum candidate logit;
* a **negative** frame must not contain any smoke-like candidate — the loss
  is applied to the mean of the three highest candidate logits.

This matches the decision rule of the detector (an alarm is raised when *any*
candidate looks like smoke) and is implemented in
`src/training/train_spatial.py` (`_mil_loss`) and
`src/training/datasets.py` (`FrameCropDataset`).

## 4. Pre-processing

| Step | Parameter | Value |
|---|---|---|
| Kalman background model | per-pixel constant velocity, 4x4 covariance | `var_init=150`, `var_process=4`, `var_measurement=25`, `var_acceleration=0.05` |
| | foreground criterion | squared distance > 9 (3 sigma) |
| | warm-up | 10 frames |
| HSV mask | saturation | <= 70 (of 255) |
| | value | >= 60 (of 255) |
| | gating | the Kalman foreground mask |
| | median filter | 5x5 |
| Morphology | open / close | 1 / 2 iterations, 3x3 ellipse |
| Components | minimum area | 0.48 % of the frame (resolution invariant) |
| | maximum area | 60 % of the frame |
| | aspect ratio | 0.6 - 8.0 |
| | maximum candidates | 8 per frame (largest first) |
| Crops | size | 50x50, RGB, scaled to [0, 1] |

## 5. Splits

Five predefined, video-level splits are generated with
`python -m src.data.video_split --num-splits 5 --seed 42`. Each split
contains 22 videos: 12 training (7 positive, 5 negative), 3 validation
(2 positive, 1 negative) and 7 test (4 positive, 3 negative). The splits are
stratified by label and grouped by recording session; the test manifests are
locked read-only and their hashes are stored in
`data/manifests/splits_locked.json`.

A **leave-one-video-out** registry (`data/manifests/lovo.csv`) is provided as
a sensitivity analysis.

## 6. Known limitations

* **Small sample.** 22 videos, 13 of them positive. A single video changes
  the video recall by 7.7 points. All conclusions are therefore reported as
  a mean and a standard deviation over the five splits, and every statistical
  test uses the video as the unit of analysis.
* **The positive videos contain almost no pre-smoke frames** (section 3.2).
* **The validation split is often uninformative for the operating point.**
  With one negative validation video that contains no candidate at all, the
  validation false-alarm estimate is identically zero and the selected
  threshold becomes extreme. This is a property of the data set size, not of
  the method, and it is the reason why the paper reports the *false alarms
  per hour versus recall* curve and compares the models at **matched recall**
  in addition to a single operating point.
* **The hard negatives are few** (3 of 9) and their assignment to the roles
  of a split is random.
* **No region-level annotation** (section 3.3).
* **The fire subset of FIRESENSE is not used** for the temporal benchmark.

## 7. D-Fire (spatial pretraining)

D-Fire is the intended source of region-level spatial supervision. The
official repository links to OneDrive and Kaggle; the OneDrive link is dead
(404) and Kaggle requires an API token, so the archive **cannot be fetched
programmatically**. `configs/data_dfire.yaml` documents the manual
acquisition procedure, the expected directory layout, the YOLO label format
and the verification steps; `src/data/download.py` records the SHA-256 of
whatever archive is present so that the manifest hash pins the exact version.

The spatial stage supports D-Fire natively (`objective: crop` with the
region-level YOLO labels). In the reported runs the spatial stage is trained
on FIRESENSE with the weakly-supervised objective, and the FIRESENSE *fire*
archive is available as an additional spatial source.

## 8. Citation

> FIRESENSE: a public benchmark for wildfire and smoke detection. Zenodo
> record 836749. <https://doi.org/10.5281/zenodo.836749>
>
> The dataset was developed within the FP7-ENVIRONMENT-244088 project
> "FIRESENSE - Fire Detection and Management through a Multi-Sensor Network
> for the Protection of Cultural Heritage Areas from the Risk of Fire and
> Extreme Weather Conditions".
