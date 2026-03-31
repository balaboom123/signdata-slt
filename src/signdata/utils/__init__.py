"""Shared utilities for the processing pipeline.

Dataset-ingestion helpers (download, class maps, frame materialisation,
availability checks) live in ``signdata.datasets._shared``.
"""

from .video import (
    FPSSampler,
    validate_video_file,
    resolve_effective_sample_fps,
)
from .files import get_video_filenames, get_filenames
from .text import normalize_text
from .manifest import (
    read_manifest,
    validate_manifest,
    has_timing,
    find_video_file,
    resolve_video_path,
    get_timing_columns,
    REQUIRED_COLUMNS,
    TIMING_COLUMNS,
    LABEL_COLUMNS,
    SPATIAL_COLUMNS,
    METADATA_COLUMNS,
    ALL_KNOWN_COLUMNS,
)
from .availability import (
    filter_available,
    AvailabilityPolicy,
)

__all__ = [
    "FPSSampler",
    "validate_video_file",
    "resolve_effective_sample_fps",
    "get_video_filenames",
    "get_filenames",
    "normalize_text",
    "read_manifest",
    "validate_manifest",
    "has_timing",
    "find_video_file",
    "resolve_video_path",
    "get_timing_columns",
    "REQUIRED_COLUMNS",
    "TIMING_COLUMNS",
    "LABEL_COLUMNS",
    "SPATIAL_COLUMNS",
    "METADATA_COLUMNS",
    "ALL_KNOWN_COLUMNS",
    "filter_available",
    "AvailabilityPolicy",
]
