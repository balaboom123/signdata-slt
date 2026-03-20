"""OpenASL dataset adapter.

OpenASL is a large-scale, open-domain ASL-English parallel corpus sourced
from YouTube.  The official release provides:

- ``openasl-v1.0.tsv``: segment manifest with columns ``vid``, ``yid``,
  ``start``, ``end``, and a text column (configurable, default ``en``)
- ``bbox-v1.0.json``: per-segment bounding boxes keyed by ``vid`` (optional)

Source config (parsed from ``config.source``):
    manifest_tsv: str              — path to openasl-v1.0.tsv
    bbox_json: str                 — path to bbox-v1.0.json (optional)
    text_column: str               — name of the translation-text column (default "en")
    availability_policy: str       — fail_fast | drop_unavailable | mark_unavailable
    download_format: str           — yt-dlp format selector
    rate_limit: str                — download rate limit
    concurrent_fragments: int      — parallel download fragments
    text_processing: dict          — text normalization options
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from .base import DatasetAdapter
from ..registry import register_dataset
from ..utils.availability import (
    AvailabilityPolicy,
    apply_availability_policy,
    get_existing_video_ids,
    write_acquire_report,
)
from ..utils.text import TextProcessingConfig, normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed source config
# ---------------------------------------------------------------------------

class OpenASLSourceConfig(BaseModel):
    """Typed config for OpenASL adapter.

    Parsed from ``config.source`` via ``get_source_config()``.

    Core columns in the official TSV are ``vid``, ``yid``, ``start``,
    ``end``.  The name of the English-text column varies across releases
    and can be set via ``text_column``.
    """

    manifest_tsv: str = ""
    bbox_json: str = ""
    text_column: str = "en"
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_format: str = (
        "worstvideo[height>=720][fps>=24]/bestvideo[height>=480]"
    )
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    text_processing: TextProcessingConfig = TextProcessingConfig()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_dataset("openasl")
class OpenASLDataset(DatasetAdapter):
    """OpenASL dataset adapter.

    Acquires YouTube videos using the ``yid`` column from the official
    TSV and builds a canonical manifest by mapping::

        vid   → SAMPLE_ID
        yid   → VIDEO_ID
        start → START
        end   → END

    Optionally merges per-segment bounding boxes from ``bbox-v1.0.json``
    into ``BBOX_X1/Y1/X2/Y2`` columns.  When bbox data is unavailable,
    use the ``detect_person`` pipeline stage instead.
    """

    name = "openasl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.source
        if not source.get("manifest_tsv"):
            raise ValueError(
                "openasl requires source.manifest_tsv pointing to "
                "the official openasl-v1.0.tsv file"
            )

    def get_source_config(self, config) -> OpenASLSourceConfig:
        return OpenASLSourceConfig(**config.source)

    # ------------------------------------------------------------------
    # acquire — download videos from YouTube
    # ------------------------------------------------------------------

    def acquire(self, config, context):
        """Download OpenASL videos from YouTube via yt-dlp."""
        source = self.get_source_config(config)
        video_dir = config.paths.videos

        if not video_dir:
            raise ValueError(
                "paths.videos is required for OpenASL. "
                "Set it in your config YAML."
            )

        os.makedirs(video_dir, exist_ok=True)

        tsv_path = source.manifest_tsv
        if not tsv_path or not Path(tsv_path).exists():
            raise FileNotFoundError(
                f"OpenASL manifest TSV not found: {tsv_path}"
            )

        tsv = pd.read_csv(tsv_path, delimiter="\t")
        if "yid" not in tsv.columns:
            raise ValueError(
                "OpenASL TSV must have a 'yid' column for YouTube video IDs. "
                f"Found columns: {list(tsv.columns)}"
            )

        all_yids = set(tsv["yid"].dropna().astype(str).unique())
        existing = get_existing_video_ids(video_dir)
        to_download = sorted(all_yids - existing)

        if not to_download:
            self.logger.info("All %d videos already downloaded.", len(all_yids))
            stats = {
                "total": len(all_yids), "downloaded": 0,
                "errors": 0, "skipped": len(all_yids),
            }
            report_dir = os.path.join(
                config.paths.root, "acquire_report",
            )
            write_acquire_report(report_dir, stats, missing=[])
            context.stats["acquire"] = stats
            return context

        self.logger.info(
            "Downloading %d / %d videos...",
            len(to_download), len(all_yids),
        )

        result = self._download_videos(to_download, video_dir, source)
        missing = result.pop("missing")
        stats = {
            "total": len(all_yids),
            "downloaded": result["downloaded"],
            "errors": result["errors"],
            "skipped": len(existing),
        }

        # Write acquire report
        report_dir = os.path.join(
            str(context.project_root), "acquire_report",
        )
        write_acquire_report(report_dir, stats, missing)

        # Enforce fail_fast at acquire time
        if source.availability_policy == "fail_fast" and result["errors"] > 0:
            raise RuntimeError(
                f"{result['errors']} download(s) failed with "
                f"availability_policy='fail_fast'. "
                f"See {report_dir}/download_report.json for details."
            )

        context.stats["acquire"] = stats
        return context

    def _download_videos(
        self,
        video_ids: List[str],
        video_dir: str,
        source: OpenASLSourceConfig,
    ) -> Dict:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import (
            DownloadError,
            ExtractorError,
            PostProcessingError,
            UnavailableVideoError,
        )

        yt_config = {
            "format": source.download_format,
            "merge_output_format": "mp4",
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
            ],
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
        missing: List[Dict] = []

        with tqdm(video_ids, desc="Downloading videos", unit="video") as pbar:
            for yid in pbar:
                time.sleep(0.2)
                url = f"https://www.youtube.com/watch?v={yid}"
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
                    self.logger.error("Error downloading %s: %s", yid, e)
                    missing.append({"VIDEO_ID": yid, "REASON": str(e)})
                    error_count += 1
                except Exception as e:
                    self.logger.error("Unexpected error for %s: %s", yid, e)
                    missing.append({"VIDEO_ID": yid, "REASON": str(e)})
                    error_count += 1
                pbar.set_postfix(errors=error_count)

        return {
            "downloaded": downloaded,
            "errors": error_count,
            "missing": missing,
        }

    # ------------------------------------------------------------------
    # build_manifest — read official TSV + optional bbox JSON
    # ------------------------------------------------------------------

    def build_manifest(self, config, context):
        """Build canonical manifest from OpenASL TSV.

        Maps ``vid`` → SAMPLE_ID, ``yid`` → VIDEO_ID, ``start`` → START,
        ``end`` → END.  The text column name is configurable via
        ``source.text_column`` (default ``"en"``).
        """
        source = self.get_source_config(config)
        manifest_path = config.paths.manifest

        tsv_path = source.manifest_tsv
        if not tsv_path or not Path(tsv_path).exists():
            raise FileNotFoundError(
                f"OpenASL manifest TSV not found: {tsv_path}"
            )

        tsv = pd.read_csv(tsv_path, delimiter="\t")

        for required in ("vid", "yid", "start", "end"):
            if required not in tsv.columns:
                raise ValueError(
                    f"OpenASL TSV missing required column '{required}'. "
                    f"Available: {list(tsv.columns)}"
                )

        text_col = source.text_column
        text_opts = source.text_processing.model_dump()
        has_text = text_col in tsv.columns

        if not has_text:
            self.logger.warning(
                "Text column '%s' not found in TSV. "
                "Manifest will have no TEXT column. "
                "Available columns: %s",
                text_col, list(tsv.columns),
            )

        # Map to canonical columns
        df = pd.DataFrame({
            "SAMPLE_ID": tsv["vid"].astype(str),
            "VIDEO_ID": tsv["yid"].astype(str),
            "START": tsv["start"].astype(float),
            "END": tsv["end"].astype(float),
        })

        if has_text:
            df["TEXT"] = (
                tsv[text_col]
                .fillna("")
                .astype(str)
                .apply(lambda t: normalize_text(t, **text_opts) if t else "")
            )

        # Pass through optional columns
        _optional_passthrough = {
            "split": "SPLIT",
            "signer_id": "SIGNER_ID",
        }
        for src_col, canon_col in _optional_passthrough.items():
            if src_col in tsv.columns:
                df[canon_col] = tsv[src_col].astype(str)

        # Merge bounding boxes if available
        if source.bbox_json and Path(source.bbox_json).exists():
            df = self._merge_bboxes(df, source.bbox_json)
            self.logger.info(
                "Merged bounding boxes from %s", source.bbox_json,
            )

        # Apply availability policy (web-mined dataset)
        video_dir = config.paths.videos
        if video_dir and Path(video_dir).is_dir():
            df = apply_availability_policy(
                df, video_dir, source.availability_policy,
            )

        # Write manifest
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        df.to_csv(manifest_path, sep="\t", index=False)

        context.manifest_path = Path(manifest_path)
        context.manifest_df = df
        context.stats["manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
        }
        self.logger.info(
            "OpenASL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), manifest_path,
        )
        return context

    @staticmethod
    def _merge_bboxes(df: pd.DataFrame, bbox_path: str) -> pd.DataFrame:
        """Merge bounding boxes from JSON into the manifest.

        Expects JSON keyed by ``vid`` (SAMPLE_ID) with values as
        ``[x1, y1, x2, y2]`` coordinate lists.  Also accepts
        ``{"bbox": [x1, y1, x2, y2]}`` dict entries.
        """
        df = df.copy()

        with open(bbox_path, "r", encoding="utf-8") as f:
            bboxes = json.load(f)

        x1, y1, x2, y2, detected = [], [], [], [], []

        for vid in df["SAMPLE_ID"]:
            bbox = bboxes.get(str(vid))
            if bbox is not None:
                # Handle both list and dict-with-bbox-key formats
                if isinstance(bbox, dict):
                    bbox = bbox.get("bbox", bbox.get("box"))
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1.append(float(bbox[0]))
                    y1.append(float(bbox[1]))
                    x2.append(float(bbox[2]))
                    y2.append(float(bbox[3]))
                    detected.append(True)
                    continue

            x1.append(None)
            y1.append(None)
            x2.append(None)
            y2.append(None)
            detected.append(False)

        df["BBOX_X1"] = x1
        df["BBOX_Y1"] = y1
        df["BBOX_X2"] = x2
        df["BBOX_Y2"] = y2
        df["PERSON_DETECTED"] = detected

        return df
