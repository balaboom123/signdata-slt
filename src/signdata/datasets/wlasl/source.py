"""WLASL source config, path resolution, and download."""

import html
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._ingestion.youtube import download_video_urls

_LEGACY_ASL_SIGNBANK_PREFIX = "https://aslsignbank.haskins.yale.edu/"
_CANONICAL_ASL_SIGNBANK_PREFIX = "https://aslsignbank.com/"
_ASL_SIGNBANK_MEDIA_PREFIX = (
    _CANONICAL_ASL_SIGNBANK_PREFIX + "dictionary/protected_media/glossvideo/"
)
_ASL_SIGNBANK_GLOSS_ID = re.compile(
    r"-(?P<gloss_id>\d+)\.(?:mp4|webm)(?:[?#].*)?$",
    re.IGNORECASE,
)
_ASL_SIGNBANK_MEDIA_URL = re.compile(
    r"(?:https://aslsignbank\.com)?"
    r"(?P<path>/dictionary/protected_media/glossvideo/[^\"'<>\s]+\.(?:mp4|webm))",
    re.IGNORECASE,
)


def normalize_download_url(url: str) -> str:
    """Map the obsolete Yale Signbank host to its certificate-valid host."""
    if url.startswith(_LEGACY_ASL_SIGNBANK_PREFIX):
        return url.replace(
            _LEGACY_ASL_SIGNBANK_PREFIX,
            _CANONICAL_ASL_SIGNBANK_PREFIX,
            1,
        )
    return url


def _fetch_asl_signbank_gloss_page(gloss_id: str) -> str:
    """Fetch the public canonical page for one ASL Signbank gloss ID."""
    url = urljoin(
        _CANONICAL_ASL_SIGNBANK_PREFIX,
        f"dictionary/gloss/{gloss_id}/",
    )
    request = Request(url, headers={"User-Agent": "signdata/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def resolve_stale_asl_signbank_url(
    url: str,
    log: logging.Logger,
) -> str:
    """Resolve a stale Signbank media path through its public gloss page.

    WLASL embeds a stable numeric Signbank gloss ID in each media filename,
    while the directory and filename around that ID can change when Signbank
    renames or reorganizes a gloss. This lookup is only used after the
    embedded URL returned 404.
    """
    normalized = normalize_download_url(url)
    if not normalized.startswith(_ASL_SIGNBANK_MEDIA_PREFIX):
        return normalized

    match = _ASL_SIGNBANK_GLOSS_ID.search(normalized)
    if match is None:
        return normalized

    gloss_id = match.group("gloss_id")
    try:
        page = _fetch_asl_signbank_gloss_page(gloss_id)
    except Exception as error:
        log.warning(
            "Could not resolve stale ASL Signbank gloss %s: %s",
            gloss_id,
            error,
        )
        return normalized

    for media_match in _ASL_SIGNBANK_MEDIA_URL.finditer(page):
        candidate = urljoin(
            _CANONICAL_ASL_SIGNBANK_PREFIX,
            html.unescape(media_match.group("path")),
        )
        if candidate.startswith(_ASL_SIGNBANK_MEDIA_PREFIX):
            if candidate != normalized:
                log.info(
                    "Resolved stale ASL Signbank media URL for gloss %s: %s",
                    gloss_id,
                    candidate,
                )
            return candidate

    log.warning(
        "ASL Signbank gloss page %s contains no public media URL",
        gloss_id,
    )
    return normalized


def _retry_stale_asl_signbank_urls(
    result: dict,
    video_dir: str,
    source: "WLASLSourceConfig",
    log: logging.Logger,
) -> dict:
    """Retry 404 Signbank media through current URLs from gloss pages."""
    if result.get("blocked"):
        return result

    retry_urls: dict[str, str] = {}
    for missing in result.get("missing", []):
        reason = str(missing.get("REASON", "")).lower()
        stale_url = str(missing.get("SOURCE_URL", ""))
        if "http error 404" not in reason:
            continue
        current_url = resolve_stale_asl_signbank_url(stale_url, log)
        if current_url != normalize_download_url(stale_url):
            retry_urls[str(missing["VIDEO_ID"])] = current_url

    if not retry_urls:
        return result

    retry_result = download_video_urls(
        retry_urls,
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
    retried_ids = set(retry_urls)
    missing = [
        item
        for item in result.get("missing", [])
        if str(item.get("VIDEO_ID")) not in retried_ids
    ]
    missing.extend(retry_result.get("missing", []))
    return {
        "downloaded": int(result.get("downloaded", 0))
        + int(retry_result.get("downloaded", 0)),
        "errors": len(missing),
        "missing": missing,
        "blocked": bool(retry_result.get("blocked", False)),
        "block_reason": str(retry_result.get("block_reason", "")),
    }


class WLASLSourceConfig(BaseModel):
    """Typed config for WLASL adapter."""

    model_config = ConfigDict(extra="forbid")

    metadata_json: str = ""
    split: Literal["all", "train", "val", "test"] = "all"
    subset: int = Field(default=0, ge=0)
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_mode: Literal["validate", "download_missing"] = "validate"
    # Capped: uncapped `bestvideo` happily pulls 4K for an isolated-sign clip.
    download_format: str = (
        "bestvideo[height>=480][height<=1080]+bestaudio/best"
    )
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    sleep_interval_requests: float = Field(default=0.0, ge=0)
    sleep_interval: float = Field(default=0.0, ge=0)
    max_sleep_interval: float | None = Field(default=None, ge=0)
    stop_on_download_block: bool = True


def iter_filtered_instances(entries: list[dict[str, Any]], source: WLASLSourceConfig):
    """Yield WLASL instances after applying subset/split filters."""
    for gloss_idx, entry in enumerate(entries):
        if source.subset and gloss_idx >= source.subset:
            break

        for inst_idx, inst in enumerate(entry.get("instances", [])):
            split = str(inst.get("split", ""))
            if source.split != "all" and split != source.split:
                continue
            yield gloss_idx, inst_idx, entry, inst


def get_source_config(config) -> WLASLSourceConfig:
    return WLASLSourceConfig(**config.dataset.source)


def read_metadata(source: WLASLSourceConfig) -> list[dict[str, Any]]:
    path = source.metadata_json
    if not path or not Path(path).exists():
        raise FileNotFoundError(
            f"WLASL metadata JSON not found: {path}\n"
            "Set dataset.source.metadata_json in your config YAML."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_release(
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

    entries = read_metadata(source)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    video_urls: dict[str, str] = {}
    for _, _, _, inst in iter_filtered_instances(entries, source):
        url = inst.get("url", "")
        video_id = inst.get("video_id", "")
        if url and video_id:
            video_urls[video_id] = normalize_download_url(str(url))

    all_ids = set(video_urls)
    existing = all_ids & get_existing_video_ids(video_dir)
    to_download_ids = sorted(all_ids - existing)

    result = {"downloaded": 0, "errors": 0, "missing": []}
    if to_download_ids:
        log.info("Downloading %d / %d videos...", len(to_download_ids), len(all_ids))
        result = download_video_urls(
            {video_id: video_urls[video_id] for video_id in to_download_ids},
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
        result = _retry_stale_asl_signbank_urls(
            result,
            video_dir,
            source,
            log,
        )
    else:
        log.info("All %d videos already downloaded.", len(all_ids))

    stats = {
        "total": len(all_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "skipped": len(existing),
    }
    report_dir = Path(config.paths.root) / "acquire_report"
    write_acquire_report(report_dir, stats, result["missing"])

    if result.get("blocked"):
        raise RuntimeError(
            "Video host blocked WLASL acquisition. Retry after a cooldown; "
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
