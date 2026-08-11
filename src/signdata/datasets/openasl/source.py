"""OpenASL source config and acquisition."""

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._ingestion.text import TextProcessingConfig
from .._ingestion.youtube import download_youtube_videos


class OpenASLSourceConfig(BaseModel):
    """Typed config for OpenASL adapter."""

    model_config = ConfigDict(extra="forbid")

    manifest_tsv: str = ""
    bbox_json: str = ""
    text_column: str = "raw-text"
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    # Same vcodec-outranks-size trap as youtube_asl; see DEFAULT_DOWNLOAD_FORMAT.
    download_format: str = (
        "worstvideo[height>=720][fps>=24][vcodec^=av01]"
        "/worstvideo[height>=720][fps>=24][vcodec^=vp09]"
        "/worstvideo[height>=720][fps>=24][vcodec^=vp9]"
        "/worstvideo[height>=720][fps>=24]"
        "/bestvideo[height>=480]"
    )
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    sleep_interval_requests: float = Field(default=0.0, ge=0)
    sleep_interval: float = Field(default=0.0, ge=0)
    max_sleep_interval: float | None = Field(default=None, ge=0)
    stop_on_download_block: bool = True
    text_processing: TextProcessingConfig = Field(
        default_factory=TextProcessingConfig
    )


def get_source_config(config) -> OpenASLSourceConfig:
    return OpenASLSourceConfig(**config.dataset.source)


def read_manifest_tsv(source: OpenASLSourceConfig) -> pd.DataFrame:
    path = source.manifest_tsv
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"OpenASL manifest TSV not found: {path}")
    return pd.read_csv(path, delimiter="\t", low_memory=False)


def download(
    source: OpenASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download OpenASL videos from YouTube via yt-dlp."""
    video_dir = config.paths.videos

    if not video_dir:
        raise ValueError(
            "paths.videos is required for OpenASL. "
            "Set it in your config YAML."
        )

    Path(video_dir).mkdir(parents=True, exist_ok=True)

    tsv = read_manifest_tsv(source)
    if "yid" not in tsv.columns:
        raise ValueError(
            "OpenASL TSV must have a 'yid' column for YouTube video IDs. "
            f"Found columns: {list(tsv.columns)}"
        )

    all_yids = set(tsv["yid"].dropna().astype(str))
    existing = all_yids & get_existing_video_ids(video_dir)
    to_download = sorted(all_yids - existing)

    result = {"downloaded": 0, "errors": 0, "missing": []}
    if to_download:
        log.info("Downloading %d / %d videos...", len(to_download), len(all_yids))
        result = download_youtube_videos(
            to_download,
            video_dir,
            download_format=source.download_format,
            rate_limit=source.rate_limit,
            concurrent_fragments=source.concurrent_fragments,
            sleep_interval_requests=source.sleep_interval_requests,
            sleep_interval=source.sleep_interval,
            max_sleep_interval=source.max_sleep_interval,
            stop_on_block=source.stop_on_download_block,
            log=log,
        )
    else:
        log.info("All %d videos already downloaded.", len(all_yids))

    stats = {
        "total": len(all_yids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "skipped": len(existing),
    }

    report_dir = Path(config.paths.root) / "acquire_report"
    write_acquire_report(report_dir, stats, result["missing"])

    if result.get("blocked"):
        raise RuntimeError(
            "Video host blocked OpenASL acquisition. Retry after a cooldown; "
            "completed files will be skipped. "
            f"See {report_dir}/missing_videos.csv for the triggering error."
        )

    if source.availability_policy == "fail_fast" and result["errors"] > 0:
        raise RuntimeError(
            f"{result['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )

    return stats
