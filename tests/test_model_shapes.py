"""Shape and causality tests of the model family."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.candidate_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    CandidateTracker,
    FeatureScaler,
    FrameFeatureBuilder,
    build_frame_features,
)
from src.models.evidence_accumulator import (
    EvidenceAccumulator,
    EvidenceConfig,
    ReliabilityModel,
)
from src.models.fixed_temporal import TemporalVoter, VotingConfig
from src.models.spatial_cnn import SpatialCNN, SpatialCNNConfig, preprocess_crops
from src.models.temporal_gru import GRUConfig, TemporalGRU
from src.models.temporal_tcn import TCNConfig, TemporalTCN


def test_spatial_cnn_output_shape():
    model = SpatialCNN(SpatialCNNConfig())
    out = model(torch.randn(8, 3, 50, 50))
    assert out.shape == (8,)
    probs = model.predict_proba(torch.randn(8, 3, 50, 50))
    assert probs.shape == (8,)
    assert bool(((probs >= 0) & (probs <= 1)).all())


def test_spatial_cnn_rejects_wrong_rank():
    model = SpatialCNN()
    with pytest.raises(ValueError):
        model(torch.randn(8, 3, 50))


def test_spatial_cnn_is_small():
    """The classifier must stay small enough for edge deployment."""
    params = sum(p.numel() for p in SpatialCNN().parameters())
    assert params < 1_000_000, f"spatial CNN has {params} parameters"


def test_preprocess_crops_normalises_and_resizes():
    crops = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    crops[0, :, :] = 255
    tensor = preprocess_crops(crops, (50, 50))
    assert tensor.shape == (3, 3, 50, 50)
    assert float(tensor.max()) <= 1.0
    assert float(tensor[0].mean()) == pytest.approx(1.0, abs=1e-3)


def test_gru_output_shapes():
    model = TemporalGRU(GRUConfig(sequence_length=16, feature_dimension=FEATURE_DIM))
    out = model(torch.randn(4, 16, FEATURE_DIM))
    assert out["now"].shape == (4,)
    assert out["horizon"].shape == (4,)
    assert out["log_variance"].shape == (4,)


def test_tcn_output_shapes():
    model = TemporalTCN(TCNConfig(sequence_length=16, input_features=FEATURE_DIM))
    out = model(torch.randn(4, 16, FEATURE_DIM))
    assert out["now"].shape == (4,)
    assert out["horizon"].shape == (4,)
    assert out["log_variance"].shape == (4,)


def test_tcn_is_causal():
    """The prediction for time t must not depend on any frame after t."""
    torch.manual_seed(0)
    model = TemporalTCN(TCNConfig(sequence_length=16, input_features=FEATURE_DIM)).eval()
    sequence = torch.randn(1, 16, FEATURE_DIM)

    # The prediction made at time k must be identical when the frames after k
    # are replaced by noise.
    k = 9
    modified = sequence.clone()
    modified[:, k + 1 :, :] = torch.randn(1, 16 - k - 1, FEATURE_DIM) * 100
    with torch.no_grad():
        base = model(sequence[:, : k + 1])["now"]
        changed = model(modified[:, : k + 1])["now"]
    assert torch.allclose(base, changed, atol=1e-5), "the TCN is not causal"

    # A change in the past *must* change the prediction. The perturbation
    # touches a single feature, because the input LayerNorm is invariant to a
    # constant offset applied to every feature of a frame.
    with torch.no_grad():
        perturbed = sequence.clone()
        perturbed[:, 0, 0] += 10.0
        assert not torch.allclose(base, model(perturbed[:, : k + 1])["now"], atol=1e-5)


def test_tcn_accepts_different_sequence_lengths():
    model = TemporalTCN(TCNConfig(sequence_length=16, input_features=FEATURE_DIM)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 32, FEATURE_DIM))
    assert out["now"].shape == (2,)


def test_feature_builder_produces_the_documented_dimension():
    builder = FrameFeatureBuilder()
    feature = builder.build(
        frame_index=0,
        candidates=[],
        cnn_probs=None,
        brightness=100.0,
        blur_score=10.0,
        foreground_ratio=0.05,
        frame_width=640,
        frame_height=480,
    )
    assert feature.shape == (FEATURE_DIM,)
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert feature[FEATURE_NAMES.index("brightness")] == pytest.approx(100.0 / 255.0, abs=1e-5)


def test_feature_scaler_round_trip():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, FEATURE_DIM)).astype(np.float32)
    scaler = FeatureScaler().fit(features)
    scaled = scaler.transform(features)
    assert abs(float(scaled.mean())) < 1e-4
    restored = FeatureScaler.from_state_dict(scaler.state_dict())
    assert np.allclose(restored.transform(features), scaled)


def test_candidate_tracker_assigns_stable_ids():
    tracker = CandidateTracker()

    class FakeCandidate:
        def __init__(self, bbox, area=100):
            self.bbox_xyxy = bbox
            self.area = area
            self.candidate_id = 0

    matches = tracker.update([FakeCandidate((0, 0, 50, 50))])
    track_id = matches[0].track_id
    matches = tracker.update([FakeCandidate((2, 2, 52, 52))])
    assert matches[0].track_id == track_id
    assert matches[0].age == 2


def test_build_frame_features_batches_frames():
    records = [
        {
            "candidates": [],
            "brightness": 50.0,
            "blur_score": 1.0,
            "foreground_ratio": 0.0,
            "frame_width": 320,
            "frame_height": 240,
        }
        for _ in range(5)
    ]
    features = build_frame_features(records)
    assert features.shape == (5, FEATURE_DIM)


def test_temporal_voter_modes():
    for mode in ("mean", "median", "majority", "max", "persistence"):
        voter = TemporalVoter(VotingConfig(mode=mode, window_seconds=1.0), fps=10.0)
        out = voter.step(0.9)
        assert "probability" in out and "decision" in out
        voter.reset()


def test_temporal_voter_persistence_requires_consecutive_frames():
    voter = TemporalVoter(
        VotingConfig(mode="persistence", confidence_threshold=0.5, persistence_seconds=0.3),
        fps=10.0,
    )
    decisions = [voter.step(0.9)["decision"] for _ in range(3)]
    assert decisions[0] is False
    assert decisions[-1] is True


def test_evidence_accumulator_hysteresis():
    # logit(0.95) = 2.94, so a threshold of 2.0 fires on the first frame.
    config = EvidenceConfig(alarm_threshold=2.0, cancel_threshold=-2.0, forgetting_factor=1.0)
    accumulator = EvidenceAccumulator(config)
    state = accumulator.step(0.95)
    assert state["evidence"] > 2.0
    assert state["state"] == "alarm"
    # A single low-probability frame must not clear the alarm.
    state = accumulator.step(0.5)
    assert state["state"] == "alarm"
    for _ in range(10):
        state = accumulator.step(0.01)
    assert state["state"] == "idle"


def test_evidence_accumulator_forgetting_factor_limits_the_evidence():
    config = EvidenceConfig(alarm_threshold=1e9, cancel_threshold=-1e9, forgetting_factor=0.5)
    accumulator = EvidenceAccumulator(config)
    for _ in range(20):
        accumulator.step(0.99)
    assert accumulator.evidence < 20.0


def test_reliability_model_outputs_probabilities():
    model = ReliabilityModel()
    features = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    values = model.predict(features)
    assert values.shape == (2,)
    assert bool(((values >= 0) & (values <= 1)).all())
