"""Pipeline-level availability utilities.

Dataset-ingestion helpers (apply_availability_policy, get_existing_video_ids,
write_acquire_report, apply_availability_policy_paths) have moved to
``signdata.datasets._shared.availability``.

This module retains:
  - ``AvailabilityPolicy`` — shared type used by both layers
  - ``filter_available``   — called by the pipeline runner after manifest load
"""

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

AvailabilityPolicy = Literal["fail_fast", "drop_unavailable", "mark_unavailable"]


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
