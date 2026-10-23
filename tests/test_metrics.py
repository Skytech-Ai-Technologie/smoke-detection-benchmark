"""Tests of the metric definitions and the statistical helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    AlarmEventRecord,
    aggregate_video_metrics,
    calibration_metrics,
    evaluate_video_events,
    frame_metrics,
    mcnemar_test,
    merge_events,
    paired_bootstrap,
    wilcoxon_signed_rank,
)


# --------------------------------------------------------------------------- #
# Frame metrics
# --------------------------------------------------------------------------- #
def test_frame_metrics_perfect_prediction():
    y_true = [0, 0, 1, 1]
    y_score = [0.0, 0.1, 0.9, 1.0]
    metrics = frame_metrics(y_true, y_score, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["auroc"] == 1.0


def test_frame_metrics_counts_confusion_matrix():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8, 0.2, 0.7]
    metrics = frame_metrics(y_true, y_score, threshold=0.5)
    assert metrics["n_true_positive"] == 2
    assert metrics["n_false_positive"] == 1
    assert metrics["n_true_negative"] == 2
    assert metrics["n_false_negative"] == 1


def test_frame_metrics_handles_a_single_class():
    metrics = frame_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert np.isnan(metrics["auroc"])
    assert metrics["n_frames"] == 3


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def test_calibration_error_is_zero_for_a_perfectly_calibrated_set():
    # Ten bins, each containing 100 samples with a probability equal to the
    # observed frequency: the expected calibration error must vanish.
    y_true = []
    y_prob = []
    for bin_index in range(10):
        p = (bin_index + 0.5) / 10.0
        positives = int(round(100 * p))
        y_true.extend([1] * positives)
        y_true.extend([0] * (100 - positives))
        y_prob.extend([p] * 100)
    metrics = calibration_metrics(np.array(y_true), np.array(y_prob), n_bins=10)
    assert metrics["ece"] < 0.02
    assert metrics["mce"] < 0.05
    # The Brier score of a *calibrated* but noisy predictor is its variance
    # (the mean of p(1-p) over the bins), not zero.
    assert 0.15 < metrics["brier"] < 0.22


def test_calibration_error_is_large_for_a_confident_wrong_model():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.99] * 100)
    metrics = calibration_metrics(y_true, y_prob)
    assert metrics["ece"] > 0.4
    assert metrics["brier"] > 0.4


def test_brier_score_of_a_perfect_forecast():
    metrics = calibration_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert metrics["brier"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Alarm events
# --------------------------------------------------------------------------- #
def test_merge_events_uses_the_refractory_period():
    events = [
        AlarmEventRecord("v", 0, 9, 0.0, 0.4),
        AlarmEventRecord("v", 20, 29, 0.8, 1.2),  # 0.4 s later -> merged
        AlarmEventRecord("v", 100, 109, 4.0, 4.4),  # far away -> separate
    ]
    merged = merge_events(events, refractory_seconds=2.0)
    assert len(merged) == 2
    assert merged[0].end_time == pytest.approx(1.2)


def test_evaluate_video_events_counts_false_alarms_on_negatives():
    mask = np.zeros(100, dtype=bool)
    mask[10:20] = True
    mask[50:60] = True
    evaluation = evaluate_video_events("neg1", "negative", 10.0, mask, None, refractory_seconds=2.0)
    assert evaluation.n_false_alarm_events == 2
    assert evaluation.n_alarm_events == 2
    assert evaluation.detected is False


def test_evaluate_video_events_computes_the_detection_delay():
    fps = 10.0
    mask = np.zeros(200, dtype=bool)
    mask[150:170] = True  # alarm starts at 15.0 s
    evaluation = evaluate_video_events("pos1", "smoke", fps, mask, onset_frame=100, refractory_seconds=2.0)
    assert evaluation.detected is True
    assert evaluation.detection_delay == pytest.approx(5.0)
    assert evaluation.recall_at_0_5s is False
    assert evaluation.recall_at_1_0s is False
    assert evaluation.recall_at_2_0s is False


def test_an_alarm_before_the_onset_is_not_a_detection():
    fps = 10.0
    mask = np.zeros(200, dtype=bool)
    mask[10:20] = True  # alarm at 1.0 s, onset at 10.0 s
    evaluation = evaluate_video_events("pos1", "smoke", fps, mask, onset_frame=100)
    assert evaluation.detected is False
    assert evaluation.detection_delay is None


def test_recall_within_the_delay_horizons():
    fps = 10.0
    mask = np.zeros(200, dtype=bool)
    mask[100:110] = True  # alarm exactly at the onset -> delay 0
    evaluation = evaluate_video_events("pos1", "smoke", fps, mask, onset_frame=100)
    assert evaluation.detected is True
    assert evaluation.detection_delay == pytest.approx(0.0)
    assert evaluation.recall_at_0_5s and evaluation.recall_at_1_0s and evaluation.recall_at_2_0s


def test_false_alarms_per_hour_normalisation():
    negatives = []
    for i in range(2):
        mask = np.zeros(100, dtype=bool)
        mask[0:10] = True
        negatives.append(
            evaluate_video_events(f"neg{i}", "negative", 10.0, mask, None, refractory_seconds=2.0)
        )
    # Two videos of 10 s each with one alarm each -> 2 events / (20/3600) h.
    metrics = aggregate_video_metrics(negatives)
    assert metrics["false_alarm_events"] == 2
    assert metrics["false_alarms_per_hour"] == pytest.approx(2 / (20.0 / 3600.0))
    assert metrics["video_specificity"] == 0.0


def test_aggregate_video_metrics_with_no_positive_videos():
    negatives = [evaluate_video_events("neg", "negative", 10.0, np.zeros(10, bool), None)]
    metrics = aggregate_video_metrics(negatives)
    assert metrics["video_recall"] == 0.0
    assert np.isnan(metrics["delay_mean"])


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_detects_a_consistent_difference():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    b = np.array([0.5, 0.55, 0.45, 0.52, 0.48, 0.5])
    result = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    assert result["mean_difference"] == pytest.approx(0.5, abs=1e-6)
    assert result["ci_lower"] > 0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_on_identical_samples():
    a = np.array([1.0, 2.0, 3.0])
    result = paired_bootstrap(a, a, n_resamples=500, seed=0)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_wilcoxon_signed_rank():
    a = np.array([1.0, 1.2, 0.9, 1.1, 1.3])
    b = np.array([0.5, 0.6, 0.4, 0.55, 0.65])
    result = wilcoxon_signed_rank(a, b)
    assert result["p_value"] < 0.1
    assert result["n_pairs"] == 5


def test_mcnemar_test():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = mcnemar_test(a, b)
    assert result["b"] == 1
    assert result["c"] == 1
    assert 0.0 <= result["p_value"] <= 1.0
