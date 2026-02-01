"""Download processor for YouTube videos and transcripts."""

import os
import time
import logging
from glob import glob
from typing import Set, Tuple, Dict

from tqdm import tqdm

from ..base import BaseProcessor
from ...registry import register_processor
from ...config.schema import Config

logger = logging.getLogger(__name__)


def _get_existing_ids(directory: str, ext: str) -> Set[str]:
    """Return set of IDs from files with the specified extension."""
    files = glob(os.path.join(directory, f"*.{ext}"))
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def _load_video_ids(file_path: str) -> Set[str]:
    """Load video IDs from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


@register_processor("download")
class DownloadProcessor(BaseProcessor):
    name = "download"

    def run(self, context):
        cfg = self.config
        video_id_file = cfg.download.video_ids_file
        transcript_dir = cfg.paths.transcripts
        video_dir = cfg.paths.videos
        languages = cfg.download.languages

        os.makedirs(transcript_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # Download transcripts
        self.logger.info("Starting transcript download...")
        transcript_stats = self._download_transcripts(
            video_id_file, transcript_dir, languages
        )

        # Download videos
        self.logger.info("Starting video download...")
        video_stats = self._download_videos(
            video_id_file, video_dir, cfg
        )

        context.stats["download"] = {
            "transcripts": transcript_stats,
            "videos": video_stats,
        }
        return context

    def _download_transcripts(
        self,
        video_id_file: str,
        transcript_dir: str,
        languages: list,
    ) -> Dict:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
        from youtube_transcript_api.formatters import JSONFormatter

        existing_ids = _get_existing_ids(transcript_dir, "json")
        all_ids = _load_video_ids(video_id_file)
        ids = list(all_ids - existing_ids)

        if not ids:
            self.logger.info("All transcripts already downloaded.")
            return {"total": len(all_ids), "downloaded": 0, "errors": 0}

        formatter = JSONFormatter()
        sleep_time = 0.2
        error_count = 0
        downloaded = 0

        with tqdm(ids, desc="Downloading transcripts") as pbar:
            for video_id in pbar:
                sleep_time = min(sleep_time, 2)
                time.sleep(sleep_time)
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, languages=languages
                    )
                    json_transcript = formatter.format_transcript(transcript)
                    path = os.path.join(transcript_dir, f"{video_id}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(json_transcript)
                    downloaded += 1
                except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                    self.logger.warning(
                        "Transcript unavailable for %s: %s", video_id, e
                    )
                    error_count += 1
                except Exception as e:
                    sleep_time += 0.1
                    self.logger.error(
                        "Error downloading transcript for %s: %s", video_id, e
                    )
                    error_count += 1
                pbar.set_postfix(errors=error_count)

        return {"total": len(all_ids), "downloaded": downloaded, "errors": error_count}

    def _download_videos(self, video_id_file: str, video_dir: str, cfg) -> Dict:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import (
            DownloadError,
            ExtractorError,
            PostProcessingError,
            UnavailableVideoError,
        )

        existing_ids = _get_existing_ids(video_dir, "mp4")
        all_ids = _load_video_ids(video_id_file)
        ids = list(all_ids - existing_ids)

        if not ids:
            self.logger.info("All videos already downloaded.")
            return {"total": len(all_ids), "downloaded": 0, "errors": 0}

        yt_config = {
            "format": cfg.download.format,
            "writesubtitles": False,
            "outtmpl": os.path.join(video_dir, "%(id)s.%(ext)s"),
            "nocheckcertificate": True,
            "geo-bypass": True,
            "limit_rate": cfg.download.rate_limit,
            "http-chunk-size": 10485760,
            "noplaylist": True,
            "no-metadata-json": True,
            "no-metadata": True,
            "concurrent-fragments": cfg.download.concurrent_fragments,
            "hls-prefer-ffmpeg": True,
            "sleep-interval": 0,
        }

        error_count = 0
        downloaded = 0

        with tqdm(ids, desc="Downloading videos", unit="video") as pbar:
            for video_id in pbar:
                time.sleep(0.2)
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                try:
                    with YoutubeDL(yt_config) as yt:
                        yt.extract_info(video_url)
                    downloaded += 1
                except (
                    DownloadError,
                    ExtractorError,
                    PostProcessingError,
                    UnavailableVideoError,
                ) as e:
                    self.logger.error("Error downloading %s: %s", video_id, e)
                    error_count += 1
                except Exception as e:
                    self.logger.error("Unexpected error for %s: %s", video_id, e)
                    error_count += 1
                pbar.set_postfix(errors=error_count)

        return {"total": len(all_ids), "downloaded": downloaded, "errors": error_count}
