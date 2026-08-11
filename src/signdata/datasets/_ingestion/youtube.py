"""Shared video download helpers using yt-dlp."""

import logging
import time
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm


class DownloadResult(TypedDict):
    downloaded: int
    errors: int
    missing: list[dict[str, str]]
    blocked: bool
    block_reason: str


_HOST_BLOCK_MARKERS = (
    "http error 429",
    "too many requests",
    "sign in to confirm you’re not a bot",
    "sign in to confirm you're not a bot",
)


def _is_host_block_error(error: Exception) -> bool:
    """Return whether yt-dlp reports host throttling or bot verification."""
    message = str(error).lower()
    return any(marker in message for marker in _HOST_BLOCK_MARKERS)


def _build_yt_config(
    video_dir: str,
    output_stem: str,
    *,
    download_format: str,
    rate_limit: int,
    concurrent_fragments: int,
    sleep_interval_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float | None = None,
) -> dict[str, object]:
    if sleep_interval_requests < 0 or sleep_interval < 0:
        raise ValueError("yt-dlp sleep intervals must be non-negative")
    if max_sleep_interval is not None and max_sleep_interval < sleep_interval:
        raise ValueError(
            "max_sleep_interval must be greater than or equal to sleep_interval"
        )

    config: dict[str, object] = {
        "format": download_format,
        "merge_output_format": "mp4",
        "postprocessors": [
            # Remuxer, not Convertor: FFmpegVideoConvertor re-encodes (it emits
            # no `-c copy`), so whenever yt-dlp fell back to .mkv because the
            # streams would not merge into mp4, it silently transcoded the
            # download to libx264 — undoing the av01/vp9 format preference and
            # costing a generation of quality. Remuxing only changes container.
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
        "outtmpl": str(Path(video_dir) / f"{output_stem}.%(ext)s"),
        "ratelimit": rate_limit,
        "http_chunk_size": 10485760,
        "noplaylist": True,
        "concurrent_fragment_downloads": concurrent_fragments,
        "sleep_interval_requests": sleep_interval_requests,
        "sleep_interval": sleep_interval,
    }
    if max_sleep_interval is not None:
        config["max_sleep_interval"] = max_sleep_interval
    return config


def _download_targets(
    targets: list[tuple[str, str]],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float | None = None,
    stop_on_block: bool = False,
    log: logging.Logger,
) -> DownloadResult:
    """Download explicit ``(video_id, url)`` targets via yt-dlp."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import parse_bytes

    parsed_rate_limit = parse_bytes(rate_limit)
    if parsed_rate_limit is None:
        raise ValueError(f"Invalid yt-dlp rate limit: {rate_limit}")

    downloaded = 0
    missing: list[dict[str, str]] = []
    block_reason = ""

    with tqdm(targets, desc="Downloading videos", unit="video") as pbar:
        for video_id, url in pbar:
            time.sleep(0.2)
            yt_config = _build_yt_config(
                video_dir,
                video_id,
                download_format=download_format,
                rate_limit=parsed_rate_limit,
                concurrent_fragments=concurrent_fragments,
                sleep_interval_requests=sleep_interval_requests,
                sleep_interval=sleep_interval,
                max_sleep_interval=max_sleep_interval,
            )
            try:
                with YoutubeDL(yt_config) as yt:
                    yt.extract_info(url, download=True)
                downloaded += 1
            except Exception as e:
                log.error("Error downloading %s from %s: %s", video_id, url, e)
                reason = str(e)
                missing.append({
                    "VIDEO_ID": video_id,
                    "SOURCE_URL": url,
                    "REASON": reason,
                })
                if stop_on_block and _is_host_block_error(e):
                    block_reason = reason
                    log.error(
                        "Video host blocked acquisition; stopping so the run can "
                        "cool down and resume without repeating completed files."
                    )
                    break
            pbar.set_postfix(errors=len(missing))

    return DownloadResult(
        downloaded=downloaded,
        errors=len(missing),
        missing=missing,
        blocked=bool(block_reason),
        block_reason=block_reason,
    )


def download_video_urls(
    video_urls: dict[str, str],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float | None = None,
    stop_on_block: bool = False,
    log: logging.Logger,
) -> DownloadResult:
    """Download videos from explicit source URLs keyed by output video ID."""
    return _download_targets(
        list(video_urls.items()),
        video_dir,
        download_format=download_format,
        rate_limit=rate_limit,
        concurrent_fragments=concurrent_fragments,
        sleep_interval_requests=sleep_interval_requests,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
        stop_on_block=stop_on_block,
        log=log,
    )


def download_youtube_videos(
    video_ids: list[str],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float | None = None,
    stop_on_block: bool = False,
    log: logging.Logger,
) -> DownloadResult:
    """Download YouTube IDs via yt-dlp."""
    return _download_targets(
        [
            (video_id, f"https://www.youtube.com/watch?v={video_id}")
            for video_id in video_ids
        ],
        video_dir,
        download_format=download_format,
        rate_limit=rate_limit,
        concurrent_fragments=concurrent_fragments,
        sleep_interval_requests=sleep_interval_requests,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
        stop_on_block=stop_on_block,
        log=log,
    )
