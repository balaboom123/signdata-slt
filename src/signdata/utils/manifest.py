"""Canonical manifest schema and shared manifest I/O utilities.

Defines the column contract that dataset adapters produce and generic
processors consume.  Provides a single ``read_manifest`` entry-point that
replaces the duplicated ``_read_manifest_csv`` helpers scattered across
extract.py, clip_video.py, webdataset.py, and detect_person.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Common video extensions produced by yt-dlp and ffmpeg
_VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".webm", ".mkv", ".avi", ".mov")

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"SAMPLE_ID", "VIDEO_ID"}

TIMING_COLUMNS = {"START", "END"}

SPLIT_COLUMNS = {"SPLIT"}

LABEL_COLUMNS = {"TEXT", "GLOSS", "CLASS_ID"}

SPATIAL_COLUMNS = {
    "BBOX_X1", "BBOX_Y1", "BBOX_X2", "BBOX_Y2", "PERSON_DETECTED",
}

METADATA_COLUMNS = {
    "SIGNER_ID", "LANGUAGE", "FPS", "VARIATION_ID", "SOURCE_URL",
}

# All known columns (union)
ALL_KNOWN_COLUMNS = (
    REQUIRED_COLUMNS
    | TIMING_COLUMNS
    | SPLIT_COLUMNS
    | LABEL_COLUMNS
    | SPATIAL_COLUMNS
    | METADATA_COLUMNS
)

# ---------------------------------------------------------------------------
# Column alias mapping — old name → canonical name
# ---------------------------------------------------------------------------

_COLUMN_ALIASES: Dict[str, str] = {
    # How2Sign / YouTube-ASL legacy names
    "SENTENCE_NAME": "SAMPLE_ID",
    "VIDEO_NAME": "VIDEO_ID",
    "START_REALIGNED": "START",
    "END_REALIGNED": "END",
    "SENTENCE": "TEXT",
    "CAPTION": "TEXT",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy column names to their canonical equivalents.

    Only renames a column if the canonical name does *not* already exist in
    the DataFrame (avoids overwriting an explicit canonical column).

    When multiple aliases map to the same canonical name (e.g. both
    ``SENTENCE`` and ``CAPTION`` → ``TEXT``), only the first alias found
    (in ``_COLUMN_ALIASES`` iteration order) is renamed.  This prevents
    ``df.rename()`` from producing duplicate column names.
    """
    rename_map = {}
    # Track which canonical names have already been claimed by a rename
    claimed_canonicals = set()
    for old_name, canonical in _COLUMN_ALIASES.items():
        if old_name in df.columns and canonical not in df.columns:
            if canonical not in claimed_canonicals:
                rename_map[old_name] = canonical
                claimed_canonicals.add(canonical)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_manifest(
    path: Union[str, Path],
    normalize_columns: bool = True,
) -> pd.DataFrame:
    """Read a TSV manifest and optionally normalize column names.

    Parameters
    ----------
    path : str or Path
        Path to the manifest TSV file.
    normalize_columns : bool
        If *True* (default), legacy column names (``VIDEO_NAME``,
        ``SENTENCE_NAME``, ``START_REALIGNED``, etc.) are automatically
        renamed to their canonical equivalents (``VIDEO_ID``,
        ``SAMPLE_ID``, ``START``, etc.).

    Returns
    -------
    pd.DataFrame
        The manifest data with (optionally) normalized column names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    df = pd.read_csv(path, delimiter="\t", on_bad_lines="skip")

    if normalize_columns:
        df = _normalize_columns(df)

    return df


def validate_manifest(df: pd.DataFrame) -> List[str]:
    """Check a manifest DataFrame for schema issues.

    Returns a list of human-readable warning/error strings.  An empty list
    means no issues were found.

    Checks performed:
    - Required columns present (SAMPLE_ID, VIDEO_ID)
    - Timing columns: if one of START/END is present, both must be
    - No duplicate SAMPLE_ID values
    """
    issues: List[str] = []

    # Required columns
    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        issues.append(
            f"Missing required columns: {sorted(missing_required)}"
        )

    # Timing column consistency
    has_start = "START" in df.columns
    has_end = "END" in df.columns
    if has_start != has_end:
        present = "START" if has_start else "END"
        missing = "END" if has_start else "START"
        issues.append(
            f"Timing column '{present}' is present but '{missing}' is "
            f"missing — both START and END are required for temporal "
            f"segmentation."
        )

    # Duplicate SAMPLE_ID
    if "SAMPLE_ID" in df.columns:
        dup_count = df["SAMPLE_ID"].duplicated().sum()
        if dup_count > 0:
            issues.append(
                f"{dup_count} duplicate SAMPLE_ID values found."
            )

    return issues


def has_timing(df: pd.DataFrame) -> bool:
    """Return True if the manifest has at least one row with both START and END non-null.

    This is used by the runner to decide whether ``clip_video`` should be
    activated (data-driven stage activation).  Requires a complete timing
    interval (both columns populated) on the *same* row — a manifest where
    START is set on one row and END on another does not qualify.
    """
    if "START" not in df.columns or "END" not in df.columns:
        return False
    return bool((df["START"].notna() & df["END"].notna()).any())


def find_video_file(
    base_dir: Union[str, Path],
    stem: str,
) -> Path:
    """Find a video file by stem, trying common video extensions.

    Tries ``.mp4`` first (most common), then other extensions.
    Falls back to ``{stem}.mp4`` if no file is found on disk, so that
    callers can rely on a deterministic return value.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing video files.
    stem : str
        File stem (e.g. a VIDEO_ID or SAMPLE_ID).

    Returns
    -------
    Path
        Path to the first matching video file, or ``base_dir/{stem}.mp4``
        as a fallback.
    """
    base_dir = Path(base_dir)
    for ext in _VIDEO_EXTENSIONS:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return base_dir / f"{stem}.mp4"


def resolve_video_path(
    row: pd.Series,
    base_dir: Union[str, Path],
) -> Path:
    """Resolve the physical video file path for a manifest row.

    Resolution order:
    1. If ``REL_PATH`` column is present and non-null → ``base_dir / REL_PATH``
    2. Otherwise → ``find_video_file(base_dir, VIDEO_ID)`` (extension-aware)

    Parameters
    ----------
    row : pd.Series
        A single manifest row.
    base_dir : str or Path
        The base directory for video files (e.g., ``config.paths.videos``
        or ``context.video_dir``).

    Returns
    -------
    Path
        Resolved absolute path to the video file.
    """
    base_dir = Path(base_dir)

    rel_path = row.get("REL_PATH") if "REL_PATH" in row.index else None
    if pd.notna(rel_path) and str(rel_path).strip():
        return base_dir / str(rel_path)

    video_id = row.get("VIDEO_ID", "")
    return find_video_file(base_dir, video_id)


def get_timing_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Return the (start_col, end_col) names present in the DataFrame.

    After ``read_manifest`` with ``normalize_columns=True``, the canonical
    names are ``START`` and ``END``.  This function also handles the legacy
    names for callers that read manifests without normalization.

    Returns
    -------
    tuple of (str, str)
        The start and end column names.

    Raises
    ------
    ValueError
        If no recognized timestamp columns are found.
    """
    columns = set(df.columns)

    if "START" in columns and "END" in columns:
        return "START", "END"
    if "START_REALIGNED" in columns and "END_REALIGNED" in columns:
        return "START_REALIGNED", "END_REALIGNED"

    raise ValueError(
        "No recognized timestamp columns found in manifest. "
        "Expected START/END or START_REALIGNED/END_REALIGNED."
    )
