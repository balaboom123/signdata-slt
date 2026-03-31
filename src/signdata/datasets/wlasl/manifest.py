"""WLASL manifest building."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .._shared.availability import apply_availability_policy
from .._shared.media import get_video_duration
from .source import WLASLSourceConfig


def build(config, source: WLASLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from WLASL_v0.3.json."""
    manifest_path = config.paths.manifest
    video_dir = config.paths.videos

    metadata_json = source.metadata_json
    if not metadata_json or not Path(metadata_json).exists():
        raise FileNotFoundError(
            f"WLASL metadata JSON not found: {metadata_json}\n"
            f"Set dataset.source.metadata_json in your config YAML."
        )

    with open(metadata_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    records = _flatten_instances(entries, video_dir)

    if not records:
        raise RuntimeError(
            "WLASL build_manifest produced no rows. "
            "Check that metadata_json is a valid WLASL JSON file."
        )

    df = pd.DataFrame(records)

    if source.subset > 0:
        df = df[df["CLASS_ID"] < source.subset].reset_index(drop=True)
        log.info("Subset WLASL-%d applied: %d rows remaining.", source.subset, len(df))

    if source.split != "all":
        df = df[df["SPLIT"] == source.split].reset_index(drop=True)
        log.info("Split filter '%s' applied: %d rows remaining.", source.split, len(df))

    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    return df


def _flatten_instances(
    entries: List[Dict],
    video_dir: Optional[str],
) -> List[Dict]:
    records = []
    for gloss_idx, entry in enumerate(entries):
        gloss = entry.get("gloss", "")
        instances = entry.get("instances", [])
        for inst_idx, inst in enumerate(instances):
            video_id = inst.get("video_id", "")
            sample_id = video_id if video_id else f"{gloss_idx}-{inst_idx}"

            fps_raw = inst.get("fps", 0)
            try:
                fps = float(fps_raw) if fps_raw else 0.0
            except (TypeError, ValueError):
                fps = 0.0

            start, end = _resolve_timing(
                video_id=video_id,
                fps=fps,
                frame_start=inst.get("frame_start"),
                frame_end=inst.get("frame_end"),
                video_dir=video_dir,
            )

            records.append({
                "SAMPLE_ID": sample_id,
                "VIDEO_ID": video_id,
                "REL_PATH": f"{video_id}.mp4",
                "SPLIT": str(inst.get("split", "")),
                "GLOSS": gloss,
                "CLASS_ID": gloss_idx,
                "SIGNER_ID": str(inst.get("signer_id", "")),
                "SOURCE_URL": str(inst.get("url", "")),
                "FPS": fps,
                "START": start,
                "END": end,
            })
    return records


def _resolve_timing(
    video_id: str,
    fps: float,
    frame_start: Any,
    frame_end: Any,
    video_dir: Optional[str],
) -> tuple:
    if frame_start is not None and frame_end is not None and fps > 0:
        try:
            return float(frame_start) / fps, float(frame_end) / fps
        except (TypeError, ValueError):
            pass

    start = 0.0
    end = 0.0
    if video_id and video_dir:
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            end = get_video_duration(video_path)
    return start, end
