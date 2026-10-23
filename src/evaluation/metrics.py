"""Evaluation metrics.

The metric definitions follow the benchmark protocol in
``paper/benchmark_protocol.md``. The most important ones are repeated here
because the paper must state them unambiguously:

``false_alarms_per_hour``
    Number of *false alarm events* divided by the total duration (in hours)
    of the **negative** videos. A false positive frame is *not* an alarm: an
    alarm event is a maximal run of alarm frames, and two runs separated by
    less than the refractory period are merged into a single event.

``detection_delay``
    For a positive video with an annotated smoke onset and at least one alarm
    that starts at or after the onset, the delay is
    ``first_alarm_time - onset_time`` in seconds. Videos without an alarm are
    *censored*: they count against video recall but not in the delay
    distribution.

``video recall``
    Fraction of positive videos in which an alarm was raised at or after the
    annotated onset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "frame_metrics",
    "calibration_metrics",
    "AlarmEventRecord",
    "VideoEvaluation",
    "evaluate_video_events",
    "aggregate_video_metrics",
    "paired_bootstrap",
    "wilcoxon_signed_rank",
    "mcnemar_test",
]


# --------------------------------------------------------------------------- #
# Frame-level metrics
# --------------------------------------------------------------------------- #
def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def frame_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    """Frame-level classification and ranking metrics.

    Args:
        y_true: Binary ground truth per frame.
        y_score: Continuous score per frame (probabilities).
        threshold: Decision threshold for the confusion-matrix metrics.

    Returns:
        A dictionary of metric values.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {}

    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = _confusion(y_true, y_pred)

    out: dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": _safe_divide(tp + tn, y_true.size),
        "specificity": _safe_divide(tn, tn + fp),
        "n_true_positive": int(tp),
        "n_false_positive": int(fp),
        "n_true_negative": int(tn),
        "n_false_negative": int(fn),
        "n_frames": int(y_true.size),
        "n_positive_frames": int(y_true.sum()),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def calibration_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> dict[str, float]:
    """Expected / maximum calibration error and the Brier score.

    The bins are equal-width over ``[0, 1]``; the reported ECE is the
    sample-weighted mean of ``|accuracy - confidence|`` per bin.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float]] = []
    total = y_true.size
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": 0})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )

    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "n_bins": n_bins,
        "bins": bins,  # type: ignore[dict-item]
    }


# --------------------------------------------------------------------------- #
# Video-level metrics
# --------------------------------------------------------------------------- #
@dataclass
class AlarmEventRecord:
    """One alarm episode in a video."""

    video_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class VideoEvaluation:
    """Per-video evaluation record."""

    video_id: str
    label: str
    fps: float
    duration_sec: float
    onset_time: float | None
    events: list[AlarmEventRecord] = field(default_factory=list)
    n_alarm_events: int = 0
    n_false_alarm_events: int = 0
    detected: bool = False
    detection_delay: float | None = None
    first_alarm_time: float | None = None
    alarm_duration_total: float = 0.0
    alarm_reset_count: int = 0
    n_alarm_frames: int = 0
    recall_at_0_5s: bool = False
    recall_at_1_0s: bool = False
    recall_at_2_0s: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "label": self.label,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "onset_time": self.onset_time,
            "n_alarm_events": self.n_alarm_events,
            "n_false_alarm_events": self.n_false_alarm_events,
            "detected": self.detected,
            "detection_delay": self.detection_delay,
            "first_alarm_time": self.first_alarm_time,
            "alarm_duration_total": self.alarm_duration_total,
            "alarm_reset_count": self.alarm_reset_count,
            "n_alarm_frames": self.n_alarm_frames,
            "recall_at_0_5s": self.recall_at_0_5s,
            "recall_at_1_0s": self.recall_at_1_0s,
            "recall_at_2_0s": self.recall_at_2_0s,
        }


def merge_events(
    events: Sequence[AlarmEventRecord],
    refractory_seconds: float,
) -> list[AlarmEventRecord]:
    """Merge alarm events separated by less than the refractory period."""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: e.start_time)
    merged = [ordered[0]]
    for event in ordered[1:]:
        last = merged[-1]
        if event.start_time - last.end_time < refractory_seconds:
            last.end_frame = max(last.end_frame, event.end_frame)
            last.end_time = max(last.end_time, event.end_time)
        else:
            merged.append(event)
    return merged


def evaluate_video_events(
    video_id: str,
    label: str,
    fps: float,
    alarm_mask: Sequence[bool],
    onset_frame: int | None,
    *,
    refractory_seconds: float = 2.0,
    total_frames: int | None = None,
) -> VideoEvaluation:
    """Turn a per-frame alarm mask into the per-video safety metrics."""
    mask = np.asarray(alarm_mask, dtype=bool).reshape(-1)
    n = int(mask.size if total_frames is None else total_frames)
    duration = n / fps if fps > 0 else 0.0
    onset_time = (onset_frame / fps) if (onset_frame is not None and fps > 0) else None

    # Extract the raw alarm runs.
    raw: list[AlarmEventRecord] = []
    start: int | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            raw.append(
                AlarmEventRecord(
                    video_id=video_id,
                    start_frame=start,
                    end_frame=i - 1,
                    start_time=start / fps,
                    end_time=(i - 1) / fps,
                )
            )
            start = None
    if start is not None:
        raw.append(
            AlarmEventRecord(
                video_id=video_id,
                start_frame=start,
                end_frame=len(mask) - 1,
                start_time=start / fps,
                end_time=(len(mask) - 1) / fps,
            )
        )

    events = merge_events(raw, refractory_seconds)

    evaluation = VideoEvaluation(
        video_id=video_id,
        label=label,
        fps=fps,
        duration_sec=duration,
        onset_time=onset_time,
        events=events,
        n_alarm_events=len(events),
        n_alarm_frames=int(mask.sum()),
        alarm_duration_total=sum(e.duration for e in events),
        alarm_reset_count=max(0, len(events) - 1),
    )

    if label == "smoke" and onset_time is not None:
        post = [e for e in events if e.start_time >= onset_time - 1e-9]
        if post:
            evaluation.detected = True
            evaluation.first_alarm_time = post[0].start_time
            delay = post[0].start_time - onset_time
            evaluation.detection_delay = float(delay)
            evaluation.recall_at_0_5s = delay <= 0.5 + 1e-9
            evaluation.recall_at_1_0s = delay <= 1.0 + 1e-9
            evaluation.recall_at_2_0s = delay <= 2.0 + 1e-9
    else:
        # Negative video: every alarm event is a false alarm.
        evaluation.n_false_alarm_events = len(events)
        if events:
            evaluation.first_alarm_time = events[0].start_time

    return evaluation


def aggregate_video_metrics(
    evaluations: Sequence[VideoEvaluation],
    refractory_seconds: float = 2.0,
) -> dict[str, Any]:
    """Aggregate per-video evaluations into the video-level metric table."""
    positives = [e for e in evaluations if e.label == "smoke"]
    negatives = [e for e in evaluations if e.label != "smoke"]

    negative_hours = sum(e.duration_sec for e in negatives) / 3600.0
    false_alarm_events = sum(e.n_false_alarm_events for e in negatives)
    delays = [e.detection_delay for e in positives if e.detection_delay is not None]

    out: dict[str, Any] = {
        "n_videos": len(evaluations),
        "n_positive_videos": len(positives),
        "n_negative_videos": len(negatives),
        "video_recall": _safe_divide(sum(1 for e in positives if e.detected), len(positives)),
        "video_specificity": _safe_divide(
            sum(1 for e in negatives if e.n_false_alarm_events == 0), len(negatives)
        ),
        "false_alarm_events": int(false_alarm_events),
        "negative_video_hours": float(negative_hours),
        "false_alarms_per_hour": _safe_divide(false_alarm_events, negative_hours),
        "alarms_per_negative_video": _safe_divide(false_alarm_events, len(negatives)),
        "n_detected_positive": sum(1 for e in positives if e.detected),
        "n_missed_positive": sum(1 for e in positives if not e.detected),
        "recall_at_0_5s": _safe_divide(sum(1 for e in positives if e.recall_at_0_5s), len(positives)),
        "recall_at_1_0s": _safe_divide(sum(1 for e in positives if e.recall_at_1_0s), len(positives)),
        "recall_at_2_0s": _safe_divide(sum(1 for e in positives if e.recall_at_2_0s), len(positives)),
        "mean_alarm_reset_count": float(np.mean([e.alarm_reset_count for e in evaluations]))
        if evaluations
        else 0.0,
        "mean_alarm_duration": float(np.mean([e.alarm_duration_total for e in evaluations]))
        if evaluations
        else 0.0,
    }

    if delays:
        arr = np.asarray(delays, dtype=np.float64)
        out.update(
            {
                "delay_mean": float(arr.mean()),
                "delay_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "delay_median": float(np.median(arr)),
                "delay_p95": float(np.percentile(arr, 95)),
                "delay_max": float(arr.max()),
                "delay_min": float(arr.min()),
                "n_delayed_videos": int(arr.size),
            }
        )
    else:
        out.update(
            {
                "delay_mean": float("nan"),
                "delay_std": float("nan"),
                "delay_median": float("nan"),
                "delay_p95": float("nan"),
                "delay_max": float("nan"),
                "delay_min": float("nan"),
                "n_delayed_videos": 0,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Statistical tests (paired, over videos)
# --------------------------------------------------------------------------- #
def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap over the *unit of analysis* (videos, not frames).

    Returns:
        The observed mean difference, its bootstrap confidence interval and
        the two-sided bootstrap p-value.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("paired bootstrap requires paired samples")
    if x.size == 0:
        return {}

    diff = x - y
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diff[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(resampled, 100 * alpha))
    upper = float(np.percentile(resampled, 100 * (1 - alpha)))
    # Two-sided bootstrap p-value for H0: mean difference = 0.
    centred = resampled - observed
    p_value = float(np.mean(np.abs(centred) >= abs(observed)))

    return {
        "mean_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "n_pairs": int(n),
        "n_resamples": int(n_resamples),
        "confidence": confidence,
    }


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test over paired per-video metrics."""
    from scipy.stats import wilcoxon

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("wilcoxon requires paired samples")
    if x.size == 0:
        return {}
    diff = x - y
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(x.size)}
    result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": int(x.size),
    }


def mcnemar_test(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """McNemar's test for paired binary outcomes (e.g. per-video detection)."""
    from scipy.stats import binomtest

    x = np.asarray(a, dtype=bool).reshape(-1)
    y = np.asarray(b, dtype=bool).reshape(-1)
    if x.size != y.size:
        raise ValueError("mcnemar requires paired samples")
    b_only = int(np.sum(x & ~y))
    c_only = int(np.sum(~x & y))
    n = b_only + c_only
    if n == 0:
        return {"b": b_only, "c": c_only, "p_value": 1.0, "n_pairs": int(x.size)}
    result = binomtest(b_only, n, 0.5, alternative="two-sided")
    return {
        "b": b_only,
        "c": c_only,
        "statistic": float(b_only),
        "p_value": float(result.pvalue),
        "n_pairs": int(x.size),
    }
