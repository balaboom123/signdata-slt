"""MS-ASL source config, path resolution, and download."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from .._shared.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._shared.youtube import download_youtube_videos

SPLITS = ("train", "val", "test")


class MSASLSourceConfig(BaseModel):
    """Typed config for MS-ASL adapter."""

    annotations_dir: str = ""
    split: str = "val"
    subset: int = 1000
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_mode: str = "validate"
    download_format: str = "bestvideo[height>=480]+bestaudio/best"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5


def get_source_config(config) -> MSASLSourceConfig:
    return MSASLSourceConfig(**config.dataset.source)


def extract_video_id(url: str) -> str:
    """Extract an 11-character YouTube video ID from a URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.split("/")[-1][:11]


def load_split_json(ann_dir: Path, split: str) -> List[Dict]:
    json_path = ann_dir / f"MSASL_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"MS-ASL annotation file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(
    source: MSASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate annotation files and video directory."""
    ann_dir = Path(source.annotations_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(f"MS-ASL annotations_dir not found: {ann_dir}")

    for split in SPLITS:
        json_path = ann_dir / f"MSASL_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"MS-ASL annotation file not found: {json_path}")

    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for MS-ASL. Set it in your config YAML."
        )
    if not Path(video_dir).exists():
        raise FileNotFoundError(f"MS-ASL video directory not found: {video_dir}")

    log.info("MS-ASL annotations validated: %s", source.annotations_dir)
    return {"validated": True}


def download_missing(
    source: MSASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download videos not already present in paths.videos."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for MS-ASL. Set it in your config YAML."
        )

    os.makedirs(video_dir, exist_ok=True)

    ann_dir = Path(source.annotations_dir)
    selected_splits = SPLITS if source.split == "all" else (source.split,)

    all_video_ids: set = set()
    for split in selected_splits:
        entries = load_split_json(ann_dir, split)
        for entry in entries:
            vid = extract_video_id(entry["url"])
            all_video_ids.add(vid)

    existing = get_existing_video_ids(video_dir)
    to_download = sorted(all_video_ids - existing)

    if not to_download:
        log.info("All %d videos already downloaded.", len(all_video_ids))
        stats = {
            "total": len(all_video_ids),
            "downloaded": 0,
            "errors": 0,
            "skipped": len(all_video_ids),
        }
        report_dir = os.path.join(config.paths.root, "acquire_report")
        write_acquire_report(report_dir, stats, missing=[])
        return stats

    log.info("Downloading %d / %d videos...", len(to_download), len(all_video_ids))

    result = download_youtube_videos(
        to_download,
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )
    missing = result.pop("missing")
    stats = {
        "total": len(all_video_ids),
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
