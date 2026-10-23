"""Model implementations: spatial front end, temporal models and the detector."""

from .calibrated_detector import (
    AlarmEvent,
    AlarmPolicy,
    AlarmPolicyConfig,
    CalibratedDetector,
    DecisionRecord,
    DetectorConfig,
    VideoResult,
)
from .calibration import (
    CALIBRATION_METHODS,
    Calibrator,
    IsotonicCalibration,
    NoCalibration,
    PlattScaling,
    TemperatureScaling,
    build_calibrator,
)
from .candidate_features import (
    FEATURE_DIM,
    FEATURE_INDEX,
    FEATURE_NAMES,
    CandidateTracker,
    FeatureScaler,
    FrameFeatureBuilder,
    build_frame_features,
)
from .evidence_accumulator import (
    EvidenceAccumulator,
    EvidenceConfig,
    ReliabilityModel,
    reliability_features,
)
from .fixed_temporal import VOTING_MODES, TemporalVoter, VotingConfig
from .kalman_hsv import (
    HSVCandidateConfig,
    KalmanBackgroundSubtractor,
    KalmanConfig,
    extract_candidates,
    image_quality,
)
from .spatial_cnn import SpatialCNN, SpatialCNNConfig, build_spatial_cnn
from .temporal_gru import GRUConfig, TemporalGRU
from .temporal_tcn import TCNConfig, TemporalTCN, build_temporal_tcn

__all__ = [
    "AlarmEvent",
    "AlarmPolicy",
    "AlarmPolicyConfig",
    "CalibratedDetector",
    "DecisionRecord",
    "DetectorConfig",
    "VideoResult",
    "CALIBRATION_METHODS",
    "Calibrator",
    "IsotonicCalibration",
    "NoCalibration",
    "PlattScaling",
    "TemperatureScaling",
    "build_calibrator",
    "FEATURE_DIM",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "CandidateTracker",
    "FeatureScaler",
    "FrameFeatureBuilder",
    "build_frame_features",
    "EvidenceAccumulator",
    "EvidenceConfig",
    "ReliabilityModel",
    "reliability_features",
    "VOTING_MODES",
    "TemporalVoter",
    "VotingConfig",
    "HSVCandidateConfig",
    "KalmanBackgroundSubtractor",
    "KalmanConfig",
    "extract_candidates",
    "image_quality",
    "SpatialCNN",
    "SpatialCNNConfig",
    "build_spatial_cnn",
    "GRUConfig",
    "TemporalGRU",
    "TCNConfig",
    "TemporalTCN",
    "build_temporal_tcn",
]
