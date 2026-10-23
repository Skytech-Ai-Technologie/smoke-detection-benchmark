"""Tests for the weakly-supervised spatial training and the operating-point
selection, the two pieces of the pipeline that were adapted to the
characteristics of the FIRESENSE data set.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import aggregate_video_metrics, evaluate_video_events
from src.models.builder import build_detector
from src.models.spatial_cnn import SpatialCNN, SpatialCNNConfig
from src.training.datasets import (
    FrameCropDataset,
    collect_frame_groups,
    frame_labels_for_video,
    mil_collate,
)
from src.training.select_thresholds import _replay_policy, select_operating_point
from src.training.train_spatial import _mil_loss, collect_crops, select_epoch_count


# --------------------------------------------------------------------------- #
# Multiple-instance objective
# --------------------------------------------------------------------------- #
def test_mil_loss_uses_the_maximum_for_positive_frames():
    """A positive frame is satisfied by a single confident candidate."""
    logits = torch.tensor([-5.0, 4.0, -5.0])
    labels = torch.tensor([1.0])
    sizes = torch.tensor([3])
    loss = _mil_loss(logits, labels, sizes)
    assert float(loss) < 0.05, "one confident candidate should satisfy a positive frame"


def test_mil_loss_rejects_a_positive_frame_without_a_confident_candidate():
    logits = torch.tensor([-4.0, -3.0, -5.0])
    labels = torch.tensor([1.0])
    sizes = torch.tensor([3])
    assert float(_mil_loss(logits, labels, sizes)) > 1.0


def test_mil_loss_rejects_a_negative_frame_with_one_confident_candidate():
    """A single outlier candidate must be enough to raise an alarm, so the
    loss has to punish it."""
    logits = torch.tensor([6.0, -4.0, -4.0, -4.0])
    labels = torch.tensor([0.0])
    sizes = torch.tensor([4])
    assert float(_mil_loss(logits, labels, sizes)) > 0.5


def test_mil_loss_accepts_a_clean_negative_frame():
    logits = torch.tensor([-6.0, -5.0, -7.0])
    labels = torch.tensor([0.0])
    sizes = torch.tensor([3])
    assert float(_mil_loss(logits, labels, sizes)) < 0.05


def test_mil_loss_handles_empty_batches():
    logits = torch.zeros(0)
    labels = torch.zeros(0)
    sizes = torch.zeros(0, dtype=torch.long)
    assert float(_mil_loss(logits, labels, sizes)) == 0.0


def test_frame_crop_dataset_and_collate():
    crops = np.zeros((3, 50, 50, 3), dtype=np.uint8)
    groups = [(crops, 1.0), (crops[:2], 0.0)]
    dataset = FrameCropDataset(groups)
    assert len(dataset) == 2
    batch = mil_collate([dataset[0], dataset[1]])
    assert batch["crops"].shape == (5, 3, 50, 50)
    assert batch["group_sizes"].tolist() == [3, 2]
    assert batch["labels"].tolist() == [1.0, 0.0]


def test_collect_frame_groups_groups_by_frame():
    class FakeVideo:
        video_id = "v"
        crops = np.zeros((4, 50, 50, 3), dtype=np.uint8)
        cand_frame_index = np.array([0, 0, 5, 5], dtype=np.int32)
        brightness = np.zeros(10, dtype=np.float32)
        frame_label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)

    groups = collect_frame_groups([FakeVideo()])
    assert len(groups) == 2
    assert groups[0][0].shape[0] == 2
    assert groups[0][1] == 0.0
    assert groups[1][1] == 1.0


# --------------------------------------------------------------------------- #
# Frame labels
# --------------------------------------------------------------------------- #
def test_frame_labels_for_a_positive_video():
    class FakeVideo:
        label = "smoke"
        onset_frame = 10
        brightness = np.zeros(20, dtype=np.float32)

    labels = frame_labels_for_video(FakeVideo(), horizon=4, guard_frames=2)
    assert labels["y_now"][10] == 1.0
    assert labels["y_now"][0] == 0.0
    assert not labels["mask_now"][9]  # the guard window is masked out
    assert labels["mask_now"][0]
    # The horizon label is an early-warning signal before the onset.
    assert labels["y_horizon"][7] == 1.0


def test_frame_labels_for_a_negative_video():
    class FakeVideo:
        label = "negative"
        onset_frame = None
        brightness = np.zeros(20, dtype=np.float32)

    labels = frame_labels_for_video(FakeVideo())
    assert labels["y_now"].sum() == 0.0
    assert labels["mask_now"].all()


# --------------------------------------------------------------------------- #
# Operating-point selection
# --------------------------------------------------------------------------- #
def test_select_operating_point_respects_the_false_alarm_budget():
    curve = [
        {"threshold": 0.5, "persistence_seconds": 0.0, "video_recall": 1.0, "false_alarms_per_hour": 50.0, "delay_mean": 1.0},
        {"threshold": 0.9, "persistence_seconds": 0.0, "video_recall": 1.0, "false_alarms_per_hour": 0.0, "delay_mean": 3.0},
        {"threshold": 0.95, "persistence_seconds": 0.0, "video_recall": 0.5, "false_alarms_per_hour": 0.0, "delay_mean": 2.0},
    ]
    selection = select_operating_point(curve, max_false_alarms_per_hour=1.0)
    assert selection["threshold"] == 0.9
    assert selection["video_recall"] == 1.0


def test_select_operating_point_falls_back_to_the_penalised_objective():
    curve = [
        {"threshold": 0.5, "persistence_seconds": 0.0, "video_recall": 1.0, "false_alarms_per_hour": 50.0, "delay_mean": 1.0},
        {"threshold": 0.9, "persistence_seconds": 0.0, "video_recall": 0.5, "false_alarms_per_hour": 30.0, "delay_mean": 3.0},
    ]
    selection = select_operating_point(curve, max_false_alarms_per_hour=1.0, false_alarm_penalty=1.0)
    assert selection["threshold"] == 0.9
    assert "no point met" in selection["selection_rule"]


def test_replay_policy_requires_the_persistence():
    probabilities = np.array([0.9, 0.9, 0.0, 0.9, 0.9, 0.9])
    uncertainties = np.zeros(6)
    reliabilities = np.zeros(6)

    no_persistence = _replay_policy(
        probabilities, uncertainties, reliabilities,
        threshold=0.5, persistence_seconds=0.0, fps=10.0,
        refractory_seconds=2.0, hysteresis_ratio=0.6, mode="fixed", max_persistence_seconds=3.0,
    )
    with_persistence = _replay_policy(
        probabilities, uncertainties, reliabilities,
        threshold=0.5, persistence_seconds=0.5, fps=10.0,
        refractory_seconds=2.0, hysteresis_ratio=0.6, mode="fixed", max_persistence_seconds=3.0,
    )
    assert no_persistence.sum() >= with_persistence.sum()
    # The isolated frame at index 2 must not raise an alarm with the
    # persistence requirement.
    assert not with_persistence[2]


def test_replay_policy_hysteresis_keeps_the_alarm_alive():
    probabilities = np.array([0.9, 0.9, 0.45, 0.9])
    mask = _replay_policy(
        probabilities, np.zeros(4), np.zeros(4),
        threshold=0.5, persistence_seconds=0.0, fps=10.0,
        refractory_seconds=0.0, hysteresis_ratio=0.6, mode="fixed", max_persistence_seconds=3.0,
    )
    # 0.45 is above the cancel threshold (0.3) so the alarm is not cleared.
    assert mask[2]


def test_select_epoch_count_uses_the_mean_curve():
    histories = [
        {"val_loss": [1.0, 0.5, 0.8]},
        {"val_loss": [2.0, 1.0, 1.5]},
    ]
    assert select_epoch_count(histories) == 2


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def test_build_detector_without_temporal_model():
    bundle = build_detector(None, "none", device="cpu")
    assert bundle.temporal_kind == "none"
    assert bundle.detector.temporal_model is None


def test_build_detector_uses_the_spatial_checkpoint(tmp_path):
    from src.utils.checkpoints import save_checkpoint

    model = SpatialCNN(SpatialCNNConfig())
    path = tmp_path / "spatial.pt"
    save_checkpoint(path, model, meta={"model_config": SpatialCNNConfig().to_dict()})
    bundle = build_detector(path, None, device="cpu")
    assert bundle.detector.spatial_model is not None
    assert bundle.detector.config.spatial_input_size == (50, 50)


def test_collect_crops_filters_the_ignore_label():
    class FakeVideo:
        crops = np.zeros((3, 50, 50, 3), dtype=np.uint8)
        cand_labels = np.array([1, 0, -1], dtype=np.int8)

    crops, labels = collect_crops([FakeVideo()])
    assert crops.shape[0] == 2
    assert labels.tolist() == [1.0, 0.0]
