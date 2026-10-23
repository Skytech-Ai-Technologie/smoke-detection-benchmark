"""Training stages: spatial pretraining, temporal training, calibration, quantisation."""

from .calibrate import calibrate_checkpoint, collect_validation_logits
from .datasets import (
    CropDataset,
    TemporalSequenceDataset,
    VideoFeatures,
    build_temporal_dataset,
    frame_features_for_video,
    load_video_features,
    score_candidates,
)
from .quantize import compare_outputs, quantize_model
from .train_spatial import collect_crops, out_of_fold_logits, train_spatial_model
from .train_temporal import fit_evidence_accumulator, prepare_features, train_neural_temporal

__all__ = [
    "calibrate_checkpoint",
    "collect_validation_logits",
    "CropDataset",
    "TemporalSequenceDataset",
    "VideoFeatures",
    "build_temporal_dataset",
    "frame_features_for_video",
    "load_video_features",
    "score_candidates",
    "compare_outputs",
    "quantize_model",
    "collect_crops",
    "out_of_fold_logits",
    "train_spatial_model",
    "fit_evidence_accumulator",
    "prepare_features",
    "train_neural_temporal",
]
