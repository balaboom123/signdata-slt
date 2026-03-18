"""YouTube-ASL dataset adapter.

Handles video/transcript acquisition from YouTube and manifest generation
from transcript JSON files.

Source config (Phase 2 bridge — reads from config.download / config.manifest):
    video_ids_file: str       — path to video ID list
    languages: list[str]      — transcript language codes
    download_format: str      — yt-dlp format selector
    rate_limit: str           — download rate limit
    concurrent_fragments: int — parallel download fragments
    max_text_length: int      — max characters per segment
    min_duration: float       — min segment duration (seconds)
    max_duration: float       — max segment duration (seconds)
    text_processing: dict     — keys: fix_encoding, normalize_whitespace,
                                lowercase, strip_punctuation
"""

import csv
import json
import os
import time
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from .base import DatasetAdapter
from ..registry import register_dataset
from ..utils.text import normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed source config
# ---------------------------------------------------------------------------

class TextProcessingConfig(BaseModel):
    """Text normalization options passed to ``normalize_text()``."""
    fix_encoding: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    strip_punctuation: bool = False


class YouTubeASLSourceConfig(BaseModel):
    """Typed config for YouTube-ASL adapter.

    In Phase 2 this is built from existing config fields via
    ``get_source_config()``.  In Phase 3 it will be parsed directly from
    ``config.source``.
    """
    video_ids_file: str = ""
    languages: List[str] = ["en"]
    download_format: str = "worstvideo[height>=720][fps>=24]/bestvideo[height>=480]"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5

    # Manifest params
    max_text_length: int = 300
    min_duration: float = 0.2
    max_duration: float = 60.0

    # Text processing
    text_processing: TextProcessingConfig = TextProcessingConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_existing_ids(directory: str, ext: str) -> Set[str]:
    """Return set of IDs from files with the specified extension."""
    files = glob(os.path.join(directory, f"*.{ext}"))
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def _load_video_ids(file_path: str) -> Set[str]:
    """Load video IDs from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_dataset("youtube_asl")
class YouTubeASLDataset(DatasetAdapter):
    name = "youtube_asl"

    @classmethod
    def validate_config(cls, config) -> None:
        if not config.download.video_ids_file:
            raise ValueError("youtube_asl requires download.video_ids_file")

    def get_source_config(self, config) -> YouTubeASLSourceConfig:
        """Build typed source config from existing config fields (Phase 2 bridge)."""
        return YouTubeASLSourceConfig(
            video_ids_file=config.download.video_ids_file,
            languages=config.download.languages,
            download_format=config.download.format,
            rate_limit=config.download.rate_limit,
            concurrent_fragments=config.download.concurrent_fragments,
            max_text_length=config.manifest.max_text_length,
            min_duration=config.manifest.min_duration,
            max_duration=config.manifest.max_duration,
            text_processing=TextProcessingConfig(
                fix_encoding=config.manifest.fix_encoding,
                normalize_whitespace=config.manifest.normalize_whitespace,
                lowercase=config.manifest.lowercase,
                strip_punctuation=config.manifest.strip_punctuation,
            ),
        )

    # ------------------------------------------------------------------
    # acquire — absorbs processors/youtube_asl/download.py
    # ------------------------------------------------------------------

    def acquire(self, config, context):
        """Download YouTube videos and transcripts."""
        source = self.get_source_config(config)
        video_dir = config.paths.videos
        transcript_dir = config.paths.transcripts

        os.makedirs(transcript_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # Download transcripts
        self.logger.info("Starting transcript download...")
        transcript_stats = self._download_transcripts(
            source.video_ids_file, transcript_dir, source.languages
        )

        # Download videos
        self.logger.info("Starting video download...")
        video_stats = self._download_videos(
            source.video_ids_file, video_dir, source
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
        languages: List[str],
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

    def _download_videos(
        self, video_id_file: str, video_dir: str, source: YouTubeASLSourceConfig
    ) -> Dict:
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
            "format": source.download_format,
            "writesubtitles": False,
            "outtmpl": os.path.join(video_dir, "%(id)s.%(ext)s"),
            "nocheckcertificate": True,
            "geo-bypass": True,
            "limit_rate": source.rate_limit,
            "http-chunk-size": 10485760,
            "noplaylist": True,
            "no-metadata-json": True,
            "no-metadata": True,
            "concurrent-fragments": source.concurrent_fragments,
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

    # ------------------------------------------------------------------
    # build_manifest — absorbs processors/youtube_asl/manifest.py
    # ------------------------------------------------------------------

    def build_manifest(self, config, context):
        """Build segmented manifest from transcript JSON files."""
        source = self.get_source_config(config)
        transcript_dir = config.paths.transcripts
        manifest_path = config.paths.manifest

        json_files = glob(os.path.join(transcript_dir, "*.json"))

        if not json_files:
            self.logger.warning("No transcript files found in %s", transcript_dir)
            context.manifest_path = Path(manifest_path)
            # Materialize empty manifest so downstream stages don't crash
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=["VIDEO_ID", "SAMPLE_ID", "START", "END", "TEXT"])
            empty_df.to_csv(manifest_path, sep="\t", index=False)
            context.manifest_df = empty_df
            context.stats["manifest"] = {"videos": 0, "segments": 0}
            return context

        self.logger.info(
            "Processing %d transcript files from %s", len(json_files), transcript_dir
        )

        # Remove existing manifest to start fresh
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        text_opts = source.text_processing.model_dump()
        processed_count = 0
        total_segments = 0
        first_write = True

        for json_file in tqdm(json_files, desc="Building manifest"):
            video_id = os.path.splitext(os.path.basename(json_file))[0]
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)

                if not transcript_data:
                    continue

                segments = self._process_segments(
                    transcript_data, video_id,
                    source.max_text_length, source.min_duration,
                    source.max_duration, text_opts,
                )

                if segments:
                    self._save_segments(segments, manifest_path, append=not first_write)
                    first_write = False
                    processed_count += 1
                    total_segments += len(segments)

            except Exception as e:
                self.logger.error("Error processing %s: %s", video_id, e)

        context.manifest_path = Path(manifest_path)
        if os.path.exists(manifest_path):
            from ..utils.manifest import read_manifest
            context.manifest_df = read_manifest(manifest_path, normalize_columns=True)

        context.stats["manifest"] = {
            "videos": processed_count,
            "segments": total_segments,
        }
        self.logger.info(
            "Manifest built: %d videos, %d segments -> %s",
            processed_count, total_segments, manifest_path,
        )
        return context

    def _process_segments(
        self,
        transcripts: List[Dict],
        video_id: str,
        max_text_length: int,
        min_duration: float,
        max_duration: float,
        text_options: Optional[Dict] = None,
    ) -> List[Dict]:
        """Parse transcript entries into filtered manifest segments.

        Produces canonical column names (VIDEO_ID, SAMPLE_ID, START, END,
        TEXT) as defined in ``utils.manifest``.
        """
        processed = []
        idx = 0

        valid = [
            t for t in transcripts
            if "text" in t and "start" in t and "duration" in t
        ]

        text_kw = text_options or {}

        for entry in valid:
            text = normalize_text(entry["text"], **text_kw)
            dur = entry["duration"]

            if (
                text
                and len(text) <= max_text_length
                and min_duration <= dur <= max_duration
            ):
                processed.append({
                    "VIDEO_ID": video_id,
                    "SAMPLE_ID": f"{video_id}-{idx:03d}",
                    "START": entry["start"],
                    "END": entry["start"] + dur,
                    "TEXT": text,
                })
                idx += 1

        return processed

    @staticmethod
    def _save_segments(segments: List[Dict], csv_path: str, append: bool = False) -> None:
        df = pd.DataFrame(segments)
        mode = "a" if append else "w"
        header = not append
        df.to_csv(
            csv_path,
            sep="\t",
            mode=mode,
            header=header,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
