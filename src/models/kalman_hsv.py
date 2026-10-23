"""Kalman background model + HSV smoke-candidate extraction.

This module implements the classical front end of the reference pipeline

    Kalman background subtraction
        -> RGB to HSV
        -> low-saturation candidate mask
        -> median filtering + connected components
        -> area / aspect-ratio filtering
        -> candidate crops and handcrafted features

The Kalman filter is a per-pixel constant-velocity tracker on the image plane.
Smoke is not a rigid object, so the filter is used as a *motion residual*
detector: pixels whose colour deviates from the tracked prediction by more
than the predicted variance are flagged as foreground, and the residual
magnitude is reused as a motion feature downstream.

The implementation is fully vectorised over pixels (float32 state and
covariance arrays), which keeps a 640x480 stream at a few tens of
milliseconds per frame on a modern CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "KalmanBackgroundSubtractor",
    "KalmanConfig",
    "HSVCandidateConfig",
    "Candidate",
    "to_hsv",
    "smoke_likelihood_mask",
    "filter_components",
    "extract_candidates",
    "image_quality",
    "DEFAULT_HSV_CONFIG",
]


@dataclass
class KalmanConfig:
    """Configuration of the per-pixel Kalman background model."""

    var_init: float = 150.0
    """Initial estimate of the per-pixel variance of the background model."""

    var_process: float = 4.0
    """Added variance of the background model (i.e. the noise of the process)."""

    var_measurement: float = 25.0
    """Variance of the measurement (i.e. of the observation noise)."""

    var_acceleration: float = 0.05
    """Variance of the (unmodelled) acceleration of the tracked pixel."""

    distance_threshold: float = 9.0
    """Squared Mahalanobis-like distance above which a pixel is foreground.

    The statistic is ``dx^2 / var_x + dy^2 / var_y``. A value of 9
    corresponds to a three-sigma deviation for a single dimension.
    """

    min_background_frames: int = 10
    """Number of frames used to initialise the background before detection."""


@dataclass
class HSVCandidateConfig:
    """Configuration of the HSV smoke-candidate stage."""

    max_saturation: int = 70
    """Upper bound on HSV saturation (0-255). Smoke is desaturated."""

    min_value: int = 60
    """Lower bound on HSV value (0-255). Smoke is not black."""

    min_area: int = 1200
    """Minimum connected-component area in pixels.

    Ignored when ``min_area_ratio`` is set. FIRESENSE mixes resolutions from
    320x240 to 704x576, so an absolute pixel threshold would not be
    comparable across videos.
    """

    min_area_ratio: float = 0.0
    """Minimum area expressed as a fraction of the frame area.

    When greater than zero this overrides ``min_area`` and makes the
    candidate filter resolution invariant.
    """

    max_area_ratio: float = 0.60
    """Reject components covering more than this fraction of the frame."""

    min_aspect_ratio: float = 0.60
    """Minimum width/height ratio of a candidate bounding box."""

    max_aspect_ratio: float = 8.00
    """Maximum width/height ratio of a candidate bounding box."""

    median_filter_ksize: int = 5
    """Kernel size of the median filter applied to the candidate mask."""

    open_iterations: int = 1
    """Morphological opening iterations applied to the candidate mask."""

    close_iterations: int = 2
    """Morphological closing iterations applied to the candidate mask."""

    crop_size: int = 50
    """Side length (in pixels) of the square crop stored for the CNN."""

    max_candidates: int = 8
    """Maximum number of candidates kept per frame (largest area first)."""


DEFAULT_HSV_CONFIG = HSVCandidateConfig()


@dataclass
class Candidate:
    """A single smoke candidate detected in one frame."""

    frame_index: int
    candidate_id: int
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    mean_hsv: tuple[float, float, float]
    motion_magnitude: float
    contour: np.ndarray | None = field(default=None, repr=False)
    crop: np.ndarray | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def aspect_ratio(self) -> float:
        h = max(self.height, 1)
        return self.width / h

    def to_record(self) -> dict[str, object]:
        """Serialise the candidate metadata (without the raw crop pixels)."""
        return {
            "frame_index": self.frame_index,
            "candidate_id": self.candidate_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "area": self.area,
            "mean_hsv": [round(v, 5) for v in self.mean_hsv],
            "motion_magnitude": round(float(self.motion_magnitude), 5),
        }


class KalmanBackgroundSubtractor:
    """Vectorised per-pixel Kalman background subtraction.

    The state of every pixel is ``[x, y, vx, vy]`` and the observation is the
    pixel intensity (the classical formulation for this family of detectors).

    The 4x4 covariance is stored as its ten unique entries so that every
    operation touches contiguous ``(H, W)`` float32 planes. This is roughly an
    order of magnitude faster than operating on a ``(H, W, 4, 4)`` array while
    being mathematically identical.
    """

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self._state: np.ndarray | None = None
        self._p: dict[str, np.ndarray] = {}
        self._shape: tuple[int, int] | None = None
        self._frames_seen = 0

    # ------------------------------------------------------------------ state
    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def initialized(self) -> bool:
        return self._shape is not None and self._frames_seen >= self.config.min_background_frames

    def reset(self) -> None:
        """Forget the background model (used when a new video starts)."""
        self._state = None
        self._p = {}
        self._shape = None
        self._frames_seen = 0

    # ------------------------------------------------------------------- api
    def apply(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the model with a frame.

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 BGR image.

        Returns:
            ``(foreground_mask, residual)`` where ``foreground_mask`` is a
            ``uint8`` 0/255 image and ``residual`` is the per-pixel absolute
            deviation used as the motion magnitude.
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR frame, got {frame_bgr.shape}")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame must be uint8")

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._shape != (h, w):
            self._init_state(h, w, gray)

        assert self._state is not None
        state = self._state

        self._predict(state)

        obs = gray
        dx = obs - state[..., 0]
        dy = obs - state[..., 1]

        var_x = np.maximum(self._p["00"], 1e-6)
        var_y = np.maximum(self._p["11"], 1e-6)
        distance = (dx * dx) / var_x + (dy * dy) / var_y

        fg = (distance > self.config.distance_threshold).astype(np.uint8) * 255
        if self._frames_seen < self.config.min_background_frames:
            fg[:] = 0

        # The motion residual must be measured against the *prediction*, i.e.
        # before the measurement update folds the current frame into the model.
        residual = np.abs(dx)

        self._update(state, obs)

        self._frames_seen += 1
        return fg, residual

    def motion_magnitude(self, residual: np.ndarray) -> float:
        """Scalar motion energy of a frame given the colour residual."""
        return float(np.mean(residual)) if residual.size else 0.0

    # -------------------------------------------------------------- internals
    def _init_state(self, h: int, w: int, gray: np.ndarray) -> None:
        cfg = self.config
        state = np.zeros((h, w, 4), dtype=np.float32)
        state[..., 0] = gray
        state[..., 1] = gray

        zeros = np.zeros((h, w), dtype=np.float32)
        p = {
            "00": np.full((h, w), cfg.var_init, dtype=np.float32),
            "01": zeros.copy(),
            "02": np.full((h, w), cfg.var_init / 2.0, dtype=np.float32),
            "03": zeros.copy(),
            "11": np.full((h, w), cfg.var_init, dtype=np.float32),
            "12": zeros.copy(),
            "13": np.full((h, w), cfg.var_init / 2.0, dtype=np.float32),
            "22": np.full((h, w), cfg.var_init, dtype=np.float32),
            "23": zeros.copy(),
            "33": np.full((h, w), cfg.var_init, dtype=np.float32),
        }
        # Symmetry: p10 = p01, p20 = p02, p30 = p03, p21 = p12, p31 = p13.
        p["10"] = p["01"]
        p["20"] = p["02"]
        p["30"] = p["03"]
        p["21"] = p["12"]
        p["31"] = p["13"]

        self._state = state
        self._p = p
        self._shape = (h, w)
        self._frames_seen = 0

    def _predict(self, state: np.ndarray) -> None:
        """Apply the constant-velocity transition in place."""
        cfg = self.config
        p = self._p

        # x' = F x
        state[..., 0] += state[..., 2]
        state[..., 1] += state[..., 3]

        # P' = F P F^T (symmetric; only the unique entries are stored).
        p00 = p["00"] + 2.0 * p["02"] + p["22"]
        p01 = p["01"] + p["03"] + p["12"] + p["23"]
        p02 = p["02"] + p["22"]
        p03 = p["03"] + p["23"]
        p11 = p["11"] + 2.0 * p["13"] + p["33"]
        p12 = p["12"] + p["23"]
        p13 = p["13"] + p["33"]
        p22 = p["22"]
        p23 = p["23"]
        p33 = p["33"]

        # Q = q [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]
        q = cfg.var_process + cfg.var_acceleration
        p00 = p00 + q
        p02 = p02 + q
        p11 = p11 + q
        p13 = p13 + q
        p22 = p22 + q
        p33 = p33 + q

        p["00"], p["01"], p["02"], p["03"] = p00, p01, p02, p03
        p["11"], p["12"], p["13"] = p11, p12, p13
        p["22"], p["23"], p["33"] = p22, p23, p33
        p["10"], p["20"], p["30"], p["21"], p["31"] = p01, p02, p03, p12, p13

    def _update(self, state: np.ndarray, obs: np.ndarray) -> None:
        """Apply the measurement update in place."""
        cfg = self.config
        p = self._p

        p00, p01, p11 = p["00"], p["01"], p["11"]

        s00 = p00 + cfg.var_measurement
        s11 = p11 + cfg.var_measurement
        s01 = p01

        det = s00 * s11 - s01 * s01
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        inv00 = s11 / det
        inv01 = -s01 / det
        inv10 = -s01 / det
        inv11 = s00 / det

        k00 = p00 * inv00 + p01 * inv01
        k01 = p00 * inv10 + p01 * inv11
        k10 = p01 * inv00 + p11 * inv01
        k11 = p01 * inv10 + p11 * inv11
        k20 = p["02"] * inv00 + p["12"] * inv01
        k21 = p["02"] * inv10 + p["12"] * inv11
        k30 = p["03"] * inv00 + p["13"] * inv01
        k31 = p["03"] * inv10 + p["13"] * inv11

        dx = obs - state[..., 0]
        dy = obs - state[..., 1]

        state[..., 0] += k00 * dx + k01 * dy
        state[..., 1] += k10 * dx + k11 * dy
        state[..., 2] += k20 * dx + k21 * dy
        state[..., 3] += k30 * dx + k31 * dy

        # P' = (I - K H) P, row i: P'[i][j] = P[i][j] - K[i,0] P[0][j] - K[i,1] P[1][j]
        n00 = p00 - k00 * p00 - k01 * p01
        n01 = p01 - k00 * p01 - k01 * p11
        n02 = p["02"] - k00 * p["02"] - k01 * p["12"]
        n03 = p["03"] - k00 * p["03"] - k01 * p["13"]
        n11 = p11 - k10 * p01 - k11 * p11
        n12 = p["12"] - k10 * p["02"] - k11 * p["12"]
        n13 = p["13"] - k10 * p["03"] - k11 * p["13"]
        n22 = p["22"] - k20 * p["02"] - k21 * p["12"]
        n23 = p["23"] - k20 * p["03"] - k21 * p["13"]
        n33 = p["33"] - k30 * p["03"] - k31 * p["13"]

        p["00"], p["01"], p["02"], p["03"] = n00, n01, n02, n03
        p["11"], p["12"], p["13"] = n11, n12, n13
        p["22"], p["23"], p["33"] = n22, n23, n33
        p["10"], p["20"], p["30"], p["21"], p["31"] = n01, n02, n03, n12, n13

    def _predicted_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Predicted background intensity for the current frame (debug helper)."""
        assert self._state is not None
        return self._state[..., 0]


def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to an 8-bit HSV image (OpenCV convention)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def smoke_likelihood_mask(
    hsv: np.ndarray,
    config: HSVCandidateConfig | None = None,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the low-saturation candidate mask.

    Args:
        hsv: 8-bit HSV image.
        foreground_mask: optional uint8 0/255 Kalman foreground mask. When
            provided, the candidate mask is restricted to moving pixels.

    Returns:
        uint8 0/255 candidate mask.
    """
    cfg = config or DEFAULT_HSV_CONFIG
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = ((sat <= cfg.max_saturation) & (val >= cfg.min_value)).astype(np.uint8) * 255

    if foreground_mask is not None:
        mask = cv2.bitwise_and(mask, foreground_mask)

    k = max(int(cfg.median_filter_ksize), 1)
    if k > 1:
        if k % 2 == 0:
            k += 1
        mask = cv2.medianBlur(mask, k)
    return mask


def filter_components(
    mask: np.ndarray,
    config: HSVCandidateConfig | None = None,
    frame_area: int | None = None,
) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    """Extract connected components and apply the geometric filters.

    Returns:
        A list of ``(contour, bbox_xyxy, area)`` sorted by decreasing area.
    """
    cfg = config or DEFAULT_HSV_CONFIG
    mask = mask.copy()

    if cfg.open_iterations > 0 and cfg.close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.open_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations)
    elif cfg.close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations)

    total_area = frame_area if frame_area is not None else mask.shape[0] * mask.shape[1]
    max_area = int(cfg.max_area_ratio * total_area)
    min_area = (
        int(round(cfg.min_area_ratio * total_area)) if cfg.min_area_ratio > 0 else int(cfg.min_area)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
            continue
        out.append((contour, (x, y, x + w, y + h), area))

    out.sort(key=lambda item: item[2], reverse=True)
    return out[: cfg.max_candidates]


def extract_candidates(
    frame_bgr: np.ndarray,
    frame_index: int,
    foreground_mask: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    config: HSVCandidateConfig | None = None,
) -> list[Candidate]:
    """Full candidate extraction for a single frame.

    Returns:
        A list of :class:`Candidate` objects sorted by decreasing area.
    """
    cfg = config or DEFAULT_HSV_CONFIG
    hsv = to_hsv(frame_bgr)
    mask = smoke_likelihood_mask(hsv, cfg, foreground_mask)
    components = filter_components(mask, cfg, frame_area=frame_bgr.shape[0] * frame_bgr.shape[1])

    candidates: list[Candidate] = []
    for cid, (contour, bbox, area) in enumerate(components):
        x0, y0, x1, y1 = bbox
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        mean_h = float(np.mean(roi[..., 0]))
        mean_s = float(np.mean(roi[..., 1]))
        mean_v = float(np.mean(roi[..., 2]))

        if residual is not None:
            patch = residual[y0:y1, x0:x1]
            motion = float(np.mean(patch)) if patch.size else 0.0
        else:
            motion = 0.0

        crop = frame_bgr[y0:y1, x0:x1]
        crop = cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)

        candidates.append(
            Candidate(
                frame_index=frame_index,
                candidate_id=cid,
                bbox_xyxy=bbox,
                area=area,
                mean_hsv=(mean_h, mean_s, mean_v),
                motion_magnitude=motion,
                contour=contour,
                crop=crop,
            )
        )
    return candidates


def image_quality(frame_bgr: np.ndarray) -> tuple[float, float]:
    """Return ``(brightness, blur_score)`` for a frame.

    ``brightness`` is the mean intensity in [0, 255]; ``blur_score`` is the
    variance of the Laplacian, a standard focus measure (higher is sharper).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur
