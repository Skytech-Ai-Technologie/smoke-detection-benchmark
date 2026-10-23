"""Data acquisition, manifest construction and feature extraction."""

from .augment import AugmentConfig, augment_batch, augment_crop
from .build_manifest import MANIFEST_COLUMNS, build_manifest, detect_label, load_annotations
from .download import DatasetFile, load_dataset_config
from .extract_frames import (
    ExtractionConfig,
    estimate_onset,
    extract_dataset,
    extract_video_features,
    load_features,
    save_features,
)
from .inspect import iter_videos, video_metadata
from .video_split import SplitConfig, build_splits, group_sessions, read_manifest, write_manifest

__all__ = [
    "AugmentConfig",
    "augment_batch",
    "augment_crop",
    "MANIFEST_COLUMNS",
    "build_manifest",
    "detect_label",
    "load_annotations",
    "DatasetFile",
    "load_dataset_config",
    "ExtractionConfig",
    "estimate_onset",
    "extract_dataset",
    "extract_video_features",
    "load_features",
    "save_features",
    "iter_videos",
    "video_metadata",
    "SplitConfig",
    "build_splits",
    "group_sessions",
    "read_manifest",
    "write_manifest",
]
