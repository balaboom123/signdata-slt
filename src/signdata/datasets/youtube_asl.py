"""YouTube-ASL dataset adapter.

Handles video/transcript acquisition from YouTube and manifest generation
from transcript JSON files.

Source config (parsed from ``config.dataset.source``):
    video_ids_file: str            — path to video ID list
    languages: list[str]           — transcript language codes
    availability_policy: str       — fail_fast | drop_unavailable | mark_unavailable
    download_format: str           — yt-dlp format selector
    rate_limit: str                — download rate limit
    concurrent_fragments: int      — parallel download fragments
    transcript_proxy_http: str     — optional HTTP proxy for transcript fetches
    transcript_proxy_https: str    — optional HTTPS proxy for transcript fetches
    stop_on_transcript_block: bool — stop transcript loop after IP blocking
    max_text_length: int           — max characters per segment
    min_duration: float            — min segment duration (seconds)
    max_duration: float            — max segment duration (seconds)
    text_processing: dict          — keys: fix_encoding, normalize_whitespace,
                                     lowercase, strip_punctuation
"""

import csv
import json
import os
import time
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from .base import DatasetAdapter
from ..registry import register_dataset
from ._shared.availability import (
    AvailabilityPolicy,
    apply_availability_policy,
    get_existing_video_ids,
    write_acquire_report,
)
from ._shared.youtube import download_youtube_videos
from ..utils.text import TextProcessingConfig, normalize_text

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPT_LANGUAGES = [
    "en",
    "ase",
    "en-US",
    "en-CA",
    "en-GB",
    "en-AU",
    "en-NZ",
    "en-IN",
    "en-ZA",
    "en-IE",
    "en-SG",
    "en-PH",
    "en-NG",
    "en-PK",
    "en-JM",
]

DEFAULT_DOWNLOAD_FORMAT = (
    "worstvideo[height>=720][fps>=24]+worstaudio"
    "/bestvideo[height>=480][height<720][fps>=24][fps<=60]+worstaudio"
    "/bestvideo[height>=480][height<=1080][fps>=14]+worstaudio"
    "/best"
)


class YouTubeASLSourceConfig(BaseModel):
    """Typed config for YouTube-ASL adapter."""
    video_ids_file: str = ""
    languages: List[str] = DEFAULT_TRANSCRIPT_LANGUAGES.copy()
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_format: str = DEFAULT_DOWNLOAD_FORMAT
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    transcript_proxy_http: Optional[str] = None
    transcript_proxy_https: Optional[str] = None
    stop_on_transcript_block: bool = True
    max_text_length: int = 300
    min_duration: float = 0.2
    max_duration: float = 60.0
    text_processing: TextProcessingConfig = TextProcessingConfig()


def _get_existing_ids(directory: str, ext: str) -> Set[str]:
    """Return set of IDs from files with the specified extension."""
    files = glob(os.path.join(directory, f"*.{ext}"))
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def _load_video_ids(file_path: str) -> Set[str]:
    """Load video IDs from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


@register_dataset("youtube_asl")
class YouTubeASLDataset(DatasetAdapter):
    name = "youtube_asl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("video_ids_file"):
            raise ValueError(
                "youtube_asl requires dataset.source.video_ids_file to be set"
            )

    def get_source_config(self, config) -> YouTubeASLSourceConfig:
        return YouTubeASLSourceConfig(**config.dataset.source)

    def download(self, config, context):
        """Download YouTube videos and transcripts."""
        source = self.get_source_config(config)
        video_dir = config.paths.videos
        transcript_dir = config.paths.transcripts

        os.makedirs(transcript_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # Download transcripts
        self.logger.info("Starting transcript download...")
        transcript_stats = self._download_transcripts(
            source.video_ids_file, transcript_dir, source
        )

        # Download videos
        self.logger.info("Starting video download...")
        video_result = self._download_videos(
            source.video_ids_file, video_dir, source
        )
        missing = video_result.pop("missing")
        video_stats = video_result

        # Write acquire report
        report_dir = os.path.join(config.paths.root, "acquire_report")
        write_acquire_report(report_dir, video_stats, missing)

        if source.availability_policy == "fail_fast" and video_stats["errors"] > 0:
            raise RuntimeError(
                f"{video_stats['errors']} download(s) failed with "
                f"availability_policy='fail_fast'. "
                f"See {report_dir}/download_report.json for details."
            )

        context.stats["dataset.download"] = {
            "transcripts": transcript_stats,
            "videos": video_stats,
        }
        return context

    def _download_transcripts(
        self,
        video_id_file: str,
        transcript_dir: str,
        source: YouTubeASLSourceConfig,
    ) -> Dict:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            IpBlocked,
            TranscriptsDisabled,
            NoTranscriptFound,
            RequestBlocked,
            VideoUnavailable,
        )
        existing_ids = _get_existing_ids(transcript_dir, "json")
        all_ids = _load_video_ids(video_id_file)
        ids = sorted(all_ids - existing_ids)

        if not ids:
            self.logger.info("All transcripts already downloaded.")
            return {
                "total": len(all_ids),
                "attempted": 0,
                "downloaded": 0,
                "errors": 0,
                "blocked": False,
            }

        sleep_time = 0.2
        error_count = 0
        downloaded = 0
        blocked = False
        proxies = self._build_transcript_proxies(source)
        transcript_client = self._build_transcript_client(source)

        with tqdm(ids, desc="Downloading transcripts") as pbar:
            for video_id in pbar:
                sleep_time = min(sleep_time, 2)
                time.sleep(sleep_time)
                try:
                    transcript = self._fetch_transcript(
                        transcript_client=transcript_client,
                        transcript_api_cls=YouTubeTranscriptApi,
                        video_id=video_id,
                        languages=source.languages,
                        proxies=proxies,
                    )
                    transcript = self._normalize_transcript_payload(transcript)
                    path = os.path.join(transcript_dir, f"{video_id}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(transcript))
                    downloaded += 1
                except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                    self.logger.warning(
                        "Transcript unavailable for %s: %s", video_id, e
                    )
                    error_count += 1
                except (RequestBlocked, IpBlocked) as e:
                    error_count += 1
                    blocked = True
                    self.logger.error(
                        "Transcript download blocked for %s: %s", video_id, e
                    )
                    if source.stop_on_transcript_block:
                        self.logger.error(
                            "Stopping transcript download early after an IP block. "
                            "Set dataset.source.transcript_proxy_http / "
                            "dataset.source.transcript_proxy_https or use a rotating "
                            "residential proxy to continue."
                        )
                        pbar.set_postfix(errors=error_count, blocked=1)
                        break
                except Exception as e:
                    sleep_time += 0.1
                    self.logger.error(
                        "Error downloading transcript for %s: %s", video_id, e
                    )
                    error_count += 1
                pbar.set_postfix(errors=error_count)

        return {
            "total": len(all_ids),
            "attempted": downloaded + error_count,
            "downloaded": downloaded,
            "errors": error_count,
            "blocked": blocked,
        }

    @staticmethod
    def _build_transcript_proxies(
        source: YouTubeASLSourceConfig,
    ) -> Optional[Dict[str, str]]:
        if not source.transcript_proxy_http and not source.transcript_proxy_https:
            return None

        return {
            "http": source.transcript_proxy_http or source.transcript_proxy_https,
            "https": source.transcript_proxy_https or source.transcript_proxy_http,
        }

    @staticmethod
    def _build_transcript_client(source: YouTubeASLSourceConfig) -> Optional[Any]:
        from youtube_transcript_api import YouTubeTranscriptApi

        proxy_config = None
        if source.transcript_proxy_http or source.transcript_proxy_https:
            from youtube_transcript_api.proxies import GenericProxyConfig

            proxy_config = GenericProxyConfig(
                http_url=source.transcript_proxy_http,
                https_url=source.transcript_proxy_https,
            )

        try:
            return YouTubeTranscriptApi(proxy_config=proxy_config)
        except TypeError:
            if proxy_config is not None:
                return None

        try:
            return YouTubeTranscriptApi()
        except TypeError:
            return None

    @staticmethod
    def _fetch_transcript(
        transcript_client: Optional[Any],
        transcript_api_cls: Any,
        video_id: str,
        languages: List[str],
        proxies: Optional[Dict[str, str]] = None,
    ) -> Any:
        if transcript_client is not None:
            if hasattr(transcript_client, "fetch"):
                return transcript_client.fetch(video_id, languages=languages)
            if hasattr(transcript_client, "list"):
                return transcript_client.list(video_id).find_transcript(
                    languages
                ).fetch()

        if hasattr(transcript_api_cls, "list_transcripts"):
            return transcript_api_cls.list_transcripts(
                video_id, proxies=proxies
            ).find_transcript(languages).fetch()

        kwargs: Dict[str, Any] = {"languages": languages}
        if proxies is not None:
            kwargs["proxies"] = proxies
        return transcript_api_cls.get_transcript(video_id, **kwargs)

    @staticmethod
    def _normalize_transcript_payload(transcript: Any) -> List[Dict]:
        if hasattr(transcript, "to_raw_data"):
            transcript = transcript.to_raw_data()

        if isinstance(transcript, list):
            return transcript

        raise TypeError(
            "Unexpected transcript payload type "
            f"{type(transcript).__name__}; expected a list or object with "
            "to_raw_data()."
        )

    def _download_videos(
        self, video_id_file: str, video_dir: str, source: YouTubeASLSourceConfig
    ) -> Dict:
        existing_ids = get_existing_video_ids(video_dir)
        all_ids = _load_video_ids(video_id_file)
        ids = list(all_ids - existing_ids)

        if not ids:
            self.logger.info("All videos already downloaded.")
            return {
                "total": len(all_ids), "downloaded": 0,
                "errors": 0, "missing": [],
            }

        result = download_youtube_videos(
            ids,
            video_dir,
            download_format=source.download_format,
            rate_limit=source.rate_limit,
            concurrent_fragments=source.concurrent_fragments,
            log=self.logger,
        )

        return {
            "total": len(all_ids),
            "downloaded": result["downloaded"],
            "errors": result["errors"],
            "missing": result["missing"],
        }

    def build_manifest(self, config, context):
        """Build segmented manifest from transcript JSON files."""
        source = self.get_source_config(config)
        transcript_dir = config.paths.transcripts
        manifest_path = config.paths.manifest

        json_files = glob(os.path.join(transcript_dir, "*.json"))

        if not json_files:
            self.logger.warning("No transcript files found in %s", transcript_dir)
            context.manifest_path = Path(manifest_path)
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=["VIDEO_ID", "SAMPLE_ID", "START", "END", "TEXT"])
            empty_df.to_csv(manifest_path, sep="\t", index=False)
            context.manifest_df = empty_df
            context.stats["dataset.manifest"] = {"videos": 0, "segments": 0}
            return context

        self.logger.info(
            "Processing %d transcript files from %s", len(json_files), transcript_dir
        )

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
            df = read_manifest(manifest_path, normalize_columns=True)

            video_dir = config.paths.videos
            if video_dir and Path(video_dir).is_dir():
                df = apply_availability_policy(
                    df, video_dir, source.availability_policy,
                )
                df.to_csv(manifest_path, sep="\t", index=False)

            context.manifest_df = df

        context.stats["dataset.manifest"] = {
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
