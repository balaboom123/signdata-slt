"""Window video processor: split long videos into fixed-length windows.

This is a **metadata-only** stage — it produces a stage manifest with
expanded window rows (new START/END per window) but does NOT produce
physical video files.  Downstream ``clip_video`` reads the windowed
manifest and creates the actual clipped segments.

Config via ``stage_config["window_video"]``::

    stage_config:
      window_video:
        window_seconds: 10.0       # length of each window
        stride_seconds: 5.0        # step between consecutive windows
        min_window_seconds: 2.0    # drop trailing windows shorter than this
        align_to_captions: false   # not yet implemented
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, field_validator

from .base import BaseProcessor
from ..registry import register_processor
from ..utils.manifest import (
    get_timing_columns,
    has_timing,
    read_manifest,
    resolve_video_path,
)

logger = logging.getLogger(__name__)

# Columns that are per-segment labels — not meaningful for arbitrary windows
_LABEL_COLUMNS = {"TEXT", "GLOSS", "CLASS_ID"}


class WindowVideoConfig(BaseModel):
    """Typed config for the window_video stage."""

    window_seconds: float = 10.0
    stride_seconds: float = 5.0
    min_window_seconds: float = 2.0
    align_to_captions: bool = False

    @field_validator("stride_seconds")
    @classmethod
    def _validate_stride(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("stride_seconds must be positive")
        return v


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def generate_windows(
    video_id: str,
    start: float,
    end: float,
    window_sec: float,
    stride_sec: float,
    min_sec: float,
    shared_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate window rows for a single video time range.

    Parameters
    ----------
    video_id : str
        Source video identifier (used to build window SAMPLE_IDs).
    start, end : float
        Time range to window over (seconds).
    window_sec : float
        Length of each window.
    stride_sec : float
        Step between consecutive window starts.
    min_sec : float
        Minimum acceptable window length (shorter trailing windows are
        dropped).
    shared_meta : dict
        Metadata columns to copy into every window row (e.g. SPLIT,
        SIGNER_ID).  ``SAMPLE_ID``, ``START``, ``END`` are overwritten.

    Returns
    -------
    list of dict
        One dict per window with VIDEO_ID, SAMPLE_ID, START, END, and
        any shared metadata.
    """
    duration = end - start
    if duration < min_sec:
        return []

    windows: List[Dict[str, Any]] = []
    t = start
    idx = 0

    while t < end:
        w_end = min(t + window_sec, end)
        w_dur = w_end - t

        if w_dur < min_sec:
            break

        row = dict(shared_meta)
        row["VIDEO_ID"] = video_id
        row["SAMPLE_ID"] = f"{video_id}_w{idx:03d}"
        row["START"] = round(t, 6)
        row["END"] = round(w_end, 6)
        windows.append(row)

        t += stride_sec
        idx += 1

    return windows


@register_processor("window_video")
class WindowVideoProcessor(BaseProcessor):
    """Split long videos into fixed-length windows (metadata only).

    Reads config from ``stage_config["window_video"]``::

        stage_config:
          window_video:
            window_seconds: 10.0
            stride_seconds: 5.0
            min_window_seconds: 2.0

    The output is a **stage manifest** at
    ``{root}/window_video/{run_name}/manifest.csv`` — no physical video
    files are created.  Downstream ``clip_video`` uses this manifest.
    """

    name = "window_video"

    def run(self, context):
        cfg = self.config
        raw_config = cfg.stage_config.get("window_video", {})
        stage_cfg = WindowVideoConfig(**raw_config)

        if stage_cfg.align_to_captions:
            self.logger.warning(
                "align_to_captions is not yet implemented; "
                "falling back to fixed-length windowing."
            )

        data = read_manifest(str(context.manifest_path), normalize_columns=True)

        # Determine time ranges per VIDEO_ID
        use_timing = has_timing(data)
        start_col = end_col = None
        if use_timing:
            start_col, end_col = get_timing_columns(data)

        # Group by VIDEO_ID to compute per-video time range
        all_windows: List[Dict[str, Any]] = []
        skipped_videos = 0

        for video_id, group in data.groupby("VIDEO_ID", sort=False):
            video_id_str = str(video_id)
            first_row = group.iloc[0]

            # Fixed-length windowing always starts at 0 and spans the full
            # video duration.  When the video file is unreadable, timed
            # manifests fall back to max(END); untimed manifests must skip.
            vid_start = 0.0
            vid_end = 0.0

            if context.video_dir:
                video_path = str(
                    resolve_video_path(first_row, str(context.video_dir))
                )
                vid_end = _get_video_duration(video_path)

            if vid_end <= 0:
                if use_timing:
                    # Fall back to caption span when video is unreadable
                    vid_end = float(group[end_col].max())
                    self.logger.warning(
                        "Cannot read video duration for %s — "
                        "using max caption END (%.1fs).",
                        video_id_str, vid_end,
                    )
                else:
                    self.logger.warning(
                        "Cannot determine duration for %s — skipping.",
                        video_id_str,
                    )
                    skipped_videos += 1
                    continue

            # Build shared metadata from the first row, dropping labels
            # and timing columns that will be overwritten
            drop = _LABEL_COLUMNS | {"SAMPLE_ID", "START", "END"}
            if start_col and start_col != "START":
                drop.add(start_col)
            if end_col and end_col != "END":
                drop.add(end_col)
            shared_meta = {
                k: v
                for k, v in first_row.to_dict().items()
                if k not in drop
            }

            windows = generate_windows(
                video_id=video_id_str,
                start=vid_start,
                end=vid_end,
                window_sec=stage_cfg.window_seconds,
                stride_sec=stage_cfg.stride_seconds,
                min_sec=stage_cfg.min_window_seconds,
                shared_meta=shared_meta,
            )
            all_windows.extend(windows)

        if not all_windows:
            raise RuntimeError(
                f"window_video produced no windows from {len(data)} manifest "
                f"rows ({skipped_videos} videos skipped). Check that video "
                f"files exist and durations exceed min_window_seconds "
                f"({stage_cfg.min_window_seconds}s)."
            )

        window_df = pd.DataFrame(all_windows)

        # Write stage manifest
        if context.stage_output_dir:
            stage_dir = context.stage_output_dir
        else:
            stage_dir = Path(cfg.paths.root) / "window_video" / cfg.run_name
        stage_dir = Path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = stage_dir / "manifest.csv"
        window_df.to_csv(manifest_path, sep="\t", index=False)
        self.logger.info(
            "Wrote %d window rows to %s", len(window_df), manifest_path,
        )

        context.manifest_df = window_df

        context.stats["window_video"] = {
            "total": len(all_windows),
            "source_rows": len(data),
            "skipped_videos": skipped_videos,
        }
        return context

    def validate_inputs(self, context) -> None:
        if not context.manifest_path or not context.manifest_path.exists():
            raise RuntimeError(
                "Cannot run window_video — manifest not found at "
                f"'{context.manifest_path}'. Run the manifest stage first."
            )
        # video_dir is only strictly required when the manifest lacks
        # timing columns (no fallback for duration).  Timed manifests
        # can fall back to max(END) when video files are unavailable.
        data = read_manifest(str(context.manifest_path), normalize_columns=True)
        if not has_timing(data):
            if not context.video_dir or not context.video_dir.exists():
                raise RuntimeError(
                    "Cannot run window_video — video directory "
                    f"'{context.video_dir}' does not exist and manifest "
                    "has no timing columns. Run upstream video stages first."
                )
