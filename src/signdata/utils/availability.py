"""Availability policy helpers for web-mined datasets.

Datasets sourced from the web (YouTube-ASL, OpenASL, WLASL) may have
videos that are deleted, region-blocked, or otherwise unavailable.
This module provides shared logic so adapters handle unavailability
consistently.

Policies:
    fail_fast          -- raise on any missing video
    drop_unavailable   -- remove rows with missing videos from manifest
    mark_unavailable   -- keep all rows, add ``AVAILABLE`` bool column
"""

import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Set

import pandas as pd

logger = logging.getLogger(__name__)

AvailabilityPolicy = Literal["fail_fast", "drop_unavailable", "mark_unavailable"]

# Extensions that yt-dlp may produce before postprocessing
_VIDEO_EXTENSIONS = ("mp4", "webm", "mkv", "avi", "mov")


def get_existing_video_ids(directory: str) -> Set[str]:
    """Return set of stem IDs from video files with any common extension."""
    ids: Set[str] = set()
    for ext in _VIDEO_EXTENSIONS:
        for f in glob(os.path.join(directory, f"*.{ext}")):
            ids.add(os.path.splitext(os.path.basename(f))[0])
    return ids


def apply_availability_policy(
    df: pd.DataFrame,
    video_dir: str,
    policy: AvailabilityPolicy,
) -> pd.DataFrame:
    """Filter or annotate manifest rows based on video availability.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest with at least a ``VIDEO_ID`` column.
    video_dir : str
        Directory where downloaded videos reside.
    policy : AvailabilityPolicy
        How to handle missing videos.

    Returns
    -------
    pd.DataFrame
        Modified manifest.

    Raises
    ------
    RuntimeError
        If *policy* is ``fail_fast`` and any VIDEO_IDs are missing.
    """
    available_ids = get_existing_video_ids(video_dir)
    is_available = df["VIDEO_ID"].isin(available_ids)
    missing_count = int((~is_available).sum())

    if missing_count == 0:
        if policy == "mark_unavailable":
            df = df.copy()
            df["AVAILABLE"] = True
        return df

    missing_ids = sorted(df.loc[~is_available, "VIDEO_ID"].unique())
    logger.warning(
        "%d rows reference %d unavailable VIDEO_IDs (policy=%s)",
        missing_count, len(missing_ids), policy,
    )

    if policy == "fail_fast":
        raise RuntimeError(
            f"{len(missing_ids)} video(s) not found in {video_dir}. "
            f"First 5: {missing_ids[:5]}. "
            f"Set availability_policy to 'drop_unavailable' or "
            f"'mark_unavailable' to continue without them."
        )

    if policy == "drop_unavailable":
        before = len(df)
        df = df[is_available].reset_index(drop=True)
        logger.info(
            "Dropped %d rows with unavailable videos (%d remaining).",
            before - len(df), len(df),
        )
    elif policy == "mark_unavailable":
        df = df.copy()
        df["AVAILABLE"] = is_available

    return df


def filter_available(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows marked unavailable (``AVAILABLE == False``).

    If the ``AVAILABLE`` column is not present, returns the DataFrame
    unchanged.  This is called by the runner after loading a manifest
    produced with ``mark_unavailable`` so that downstream processors
    only iterate over rows with actual video files on disk.
    """
    if "AVAILABLE" not in df.columns:
        return df
    filtered = df[df["AVAILABLE"]].reset_index(drop=True)
    n_dropped = len(df) - len(filtered)
    if n_dropped:
        logger.info(
            "Filtered %d unavailable rows from manifest (%d remaining).",
            n_dropped, len(filtered),
        )
    return filtered


def write_acquire_report(
    report_dir: str,
    stats: Dict,
    missing: List[Dict],
) -> None:
    """Write acquire report files.

    Parameters
    ----------
    report_dir : str
        Directory for report files (created if needed).
    stats : dict
        Summary stats (total, downloaded, errors, skipped).
    missing : list of dict
        Each entry has ``VIDEO_ID`` and ``REASON`` keys.
    """
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "download_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    missing_path = os.path.join(report_dir, "missing_videos.csv")
    if missing:
        pd.DataFrame(missing).to_csv(missing_path, index=False)
    else:
        # Write empty CSV with headers
        pd.DataFrame(columns=["VIDEO_ID", "REASON"]).to_csv(
            missing_path, index=False,
        )
