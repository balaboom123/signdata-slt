"""Shared YouTube video download helpers using yt-dlp.

Provides a single ``download_youtube_videos`` function that eliminates
duplicated yt-dlp boilerplate across dataset adapters (OpenASL, YouTube-ASL,
MS-ASL, WLASL).
"""

import logging
import os
import time
from typing import Dict, List, Optional, TypedDict

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadResult(TypedDict):
    downloaded: int
    errors: int
    missing: List[Dict[str, str]]


def download_youtube_videos(
    video_ids: List[str],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> DownloadResult:
    """Download YouTube videos via yt-dlp.

    Parameters
    ----------
    video_ids : list[str]
        YouTube video IDs to download.
    video_dir : str
        Directory to save downloaded videos.
    download_format : str
        yt-dlp format selector string.
    rate_limit : str
        Download rate limit (e.g. ``"5M"``).
    concurrent_fragments : int
        Number of parallel download fragments.
    sleep_interval : float
        Seconds to wait between downloads.
    log : logging.Logger, optional
        Logger instance. Falls back to module-level logger.

    Returns
    -------
    DownloadResult
        Dict with ``downloaded``, ``errors``, and ``missing`` keys.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import (
        DownloadError,
        ExtractorError,
        PostProcessingError,
        UnavailableVideoError,
    )

    _log = log or logger

    yt_config = {
        "format": download_format,
        "merge_output_format": "mp4",
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "writesubtitles": False,
        "outtmpl": os.path.join(video_dir, "%(id)s.%(ext)s"),
        "nocheckcertificate": True,
        "geo-bypass": True,
        "limit_rate": rate_limit,
        "http-chunk-size": 10485760,
        "noplaylist": True,
        "no-metadata-json": True,
        "no-metadata": True,
        "concurrent-fragments": concurrent_fragments,
        "hls-prefer-ffmpeg": True,
        "sleep-interval": 0,
    }

    error_count = 0
    downloaded = 0
    missing: List[Dict[str, str]] = []

    with tqdm(video_ids, desc="Downloading videos", unit="video") as pbar:
        for video_id in pbar:
            time.sleep(sleep_interval)
            url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                with YoutubeDL(yt_config) as yt:
                    yt.extract_info(url)
                downloaded += 1
            except (
                DownloadError,
                ExtractorError,
                PostProcessingError,
                UnavailableVideoError,
            ) as e:
                _log.error("Error downloading %s: %s", video_id, e)
                missing.append({"VIDEO_ID": video_id, "REASON": str(e)})
                error_count += 1
            except Exception as e:
                _log.error("Unexpected error for %s: %s", video_id, e)
                missing.append({"VIDEO_ID": video_id, "REASON": str(e)})
                error_count += 1
            pbar.set_postfix(errors=error_count)

    return DownloadResult(
        downloaded=downloaded,
        errors=error_count,
        missing=missing,
    )
