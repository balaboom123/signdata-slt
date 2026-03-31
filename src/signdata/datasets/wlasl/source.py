"""WLASL source config, path resolution, and download."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from .._shared.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._shared.youtube import download_youtube_videos


class WLASLSourceConfig(BaseModel):
    """Typed config for WLASL adapter."""

    metadata_json: str = ""
    split: str = "all"
    subset: int = 0
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_mode: str = "validate"
    download_format: str = "bestvideo[height>=480]+bestaudio/best"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5


def get_source_config(config) -> WLASLSourceConfig:
    return WLASLSourceConfig(**config.dataset.source)


def validate_release(
    source: WLASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate that the video directory exists."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for WLASL. Set it in your config YAML."
        )
    video_path = Path(video_dir)
    if not video_path.exists():
        raise FileNotFoundError(
            f"WLASL video directory not found: {video_dir}\n"
            f"Set paths.videos to the directory containing downloaded WLASL video files."
        )
    existing = get_existing_video_ids(video_dir)
    log.info(
        "WLASL video directory validated: %s (%d videos found)",
        video_dir, len(existing),
    )
    return {"validated": True, "videos_on_disk": len(existing)}


def download_missing(
    source: WLASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download any WLASL videos missing from disk."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for WLASL. Set it in your config YAML."
        )

    metadata_json = source.metadata_json
    if not metadata_json or not Path(metadata_json).exists():
        raise FileNotFoundError(f"WLASL metadata JSON not found: {metadata_json}")

    os.makedirs(video_dir, exist_ok=True)

    with open(metadata_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    url_to_id: Dict[str, str] = {}
    for entry in entries:
        for inst in entry.get("instances", []):
            url = inst.get("url", "")
            video_id = inst.get("video_id", "")
            if url and video_id:
                url_to_id[video_id] = url

    all_ids = set(url_to_id.keys())
    existing = get_existing_video_ids(video_dir)
    to_download_ids = sorted(all_ids - existing)

    if not to_download_ids:
        log.info("All %d videos already downloaded.", len(all_ids))
        stats = {"total": len(all_ids), "downloaded": 0, "errors": 0, "skipped": len(all_ids)}
        report_dir = os.path.join(config.paths.root, "acquire_report")
        write_acquire_report(report_dir, stats, missing=[])
        return stats

    log.info("Downloading %d / %d videos...", len(to_download_ids), len(all_ids))

    result = download_youtube_videos(
        to_download_ids,
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )
    missing = result["missing"]
    stats = {
        "total": len(all_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "skipped": len(existing),
    }
    report_dir = os.path.join(config.paths.root, "acquire_report")
    write_acquire_report(report_dir, stats, missing)

    if source.availability_policy == "fail_fast" and result["errors"] > 0:
        raise RuntimeError(
            f"{result['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )
    return stats
