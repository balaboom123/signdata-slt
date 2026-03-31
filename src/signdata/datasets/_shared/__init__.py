"""Shared helpers for dataset ingestion (download + manifest stages).

These modules are used exclusively by dataset adapters.  Generic pipeline
utilities (FPS sampling, manifest reading, etc.) live in ``signdata.utils``.

Submodules
----------
availability    -- AvailabilityPolicy, existence checks, policy enforcement
classmap        -- TSV class-map loading and joining
media           -- video duration/FPS probing; frame-sequence materialisation
paths           -- path resolution helpers
youtube         -- yt-dlp download wrapper
"""

from .availability import (
    AvailabilityPolicy,
    apply_availability_policy,
    apply_availability_policy_paths,
    get_existing_video_ids,
    write_acquire_report,
)
from .classmap import join_class_map, load_class_map
from .media import get_video_duration, get_video_fps, materialize_frames_to_video
from .paths import resolve_dir
from .youtube import DownloadResult, download_youtube_videos

__all__ = [
    "AvailabilityPolicy",
    "apply_availability_policy",
    "apply_availability_policy_paths",
    "get_existing_video_ids",
    "write_acquire_report",
    "join_class_map",
    "load_class_map",
    "get_video_duration",
    "get_video_fps",
    "materialize_frames_to_video",
    "resolve_dir",
    "DownloadResult",
    "download_youtube_videos",
]
