"""Tests for the Kalman background model and the HSV candidate extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.kalman_hsv import (
    HSVCandidateConfig,
    KalmanBackgroundSubtractor,
    KalmanConfig,
    extract_candidates,
    filter_components,
    image_quality,
    smoke_likelihood_mask,
    to_hsv,
)


def _static_video(n_frames: int = 25, h: int = 120, w: int = 160) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    return [base.copy() for _ in range(n_frames)]


def test_kalman_converges_to_empty_foreground_on_a_static_scene():
    """A static scene must not produce foreground once the model has converged."""
    sub = KalmanBackgroundSubtractor(KalmanConfig(min_background_frames=5))
    frames = _static_video(30)
    for frame in frames[:-1]:
        fg, _ = sub.apply(frame)
    fg, _ = sub.apply(frames[-1])
    assert fg.shape == frames[-1].shape[:2]
    assert fg.dtype == np.uint8
    assert (fg > 0).mean() < 0.01


def test_kalman_flags_a_moving_object():
    """A moving bright square must appear in the foreground mask."""
    sub = KalmanBackgroundSubtractor(KalmanConfig(min_background_frames=5))
    frames = _static_video(30)
    for frame in frames:
        sub.apply(frame)
    moving = frames[-1].copy()
    moving[40:80, 50:90] = 255
    fg, residual = sub.apply(moving)
    assert (fg > 0).mean() > 0.01
    assert residual[40:80, 50:90].mean() > residual.mean()


def test_kalman_resets_cleanly():
    sub = KalmanBackgroundSubtractor(KalmanConfig(min_background_frames=3))
    frames = _static_video(10)
    for frame in frames:
        sub.apply(frame)
    assert sub.frames_seen == 10
    sub.reset()
    assert sub.frames_seen == 0
    assert not sub.initialized


def test_kalman_rejects_invalid_input():
    sub = KalmanBackgroundSubtractor()
    with pytest.raises(ValueError):
        sub.apply(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        sub.apply(np.zeros((10, 10, 3), dtype=np.float32))


def test_hsv_mask_selects_desaturated_bright_pixels():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[10:20, 10:20] = (200, 200, 200)  # grey: low saturation, high value
    frame[25:35, 25:35] = (0, 0, 255)  # red: high saturation
    hsv = to_hsv(frame)
    mask = smoke_likelihood_mask(hsv, HSVCandidateConfig(min_value=60, max_saturation=70))
    assert mask[15, 15] > 0
    assert mask[30, 30] == 0


def test_candidate_extraction_respects_the_area_filter():
    config = HSVCandidateConfig(min_area=500, min_area_ratio=0.0, max_candidates=8)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:70, 50:70] = 180  # 20x20 = 400 px -> below the threshold
    candidates = extract_candidates(frame, 0, config=config)
    assert candidates == []

    frame[40:90, 40:90] = 180  # 50x50 = 2500 px -> above the threshold
    candidates = extract_candidates(frame, 1, config=config)
    assert len(candidates) == 1
    assert candidates[0].area >= 500
    assert candidates[0].crop.shape == (config.crop_size, config.crop_size, 3)


def test_candidate_extraction_is_resolution_invariant():
    """A ratio-based area threshold must accept the same object at two scales."""
    config = HSVCandidateConfig(min_area_ratio=0.01, min_area=0)
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    small[20:50, 20:50] = 180  # 30x30 = 900 px = 9 % of the frame
    large = np.zeros((200, 200, 3), dtype=np.uint8)
    large[40:100, 40:100] = 180  # 60x60 = 3600 px = 9 % of the frame
    assert len(extract_candidates(small, 0, config=config)) == 1
    assert len(extract_candidates(large, 0, config=config)) == 1


def test_filter_components_rejects_extreme_aspect_ratios():
    config = HSVCandidateConfig(min_area=100, min_aspect_ratio=0.6, max_aspect_ratio=8.0)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:20, 10:190] = 255  # 180x10 -> aspect ratio 18
    components = filter_components(mask, config, frame_area=20000)
    assert components == []


def test_image_quality_reports_brightness_and_blur():
    flat = np.full((40, 40, 3), 128, dtype=np.uint8)
    brightness, blur = image_quality(flat)
    assert 120 <= brightness <= 136
    assert blur < 1.0

    noisy = np.random.default_rng(1).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    _, sharp = image_quality(noisy)
    assert sharp > blur
