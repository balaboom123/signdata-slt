"""LSA64 source config, path resolution, and release validation."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from .._shared.availability import AvailabilityPolicy
from .._shared.classmap import load_class_map

_BUNDLED_CLASS_MAP = (
    Path(__file__).parent.parent.parent.parent.parent / "assets" / "lsa64_class_map.tsv"
)

DEFAULT_FPS = 60.0


class LSA64SourceConfig(BaseModel):
    """Typed config for LSA64 adapter."""

    release_dir: str = ""
    variant: str = "cut"
    split: str = "all"
    split_strategy: str = "none"
    train_signers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    val_signers: List[int] = [9]
    test_signers: List[int] = [10]
    class_map_file: str = ""
    availability_policy: AvailabilityPolicy = "fail_fast"
    allow_missing_class_map: bool = False


def get_source_config(config) -> LSA64SourceConfig:
    return LSA64SourceConfig(**dict(config.dataset.source))


def resolve_video_dir(config, source: LSA64SourceConfig) -> str:
    if source.release_dir:
        return source.release_dir
    return config.paths.videos or ""


def validate_release(
    source: LSA64SourceConfig,
    video_dir: str,
    log: logging.Logger,
) -> dict:
    """Validate LSA64 release directory. Returns stats dict."""
    if not video_dir:
        raise FileNotFoundError(
            "LSA64 requires a local release directory. "
            "Set dataset.source.release_dir or paths.videos in your config YAML.\n"
            "Download LSA64 from https://facundoq.github.io/datasets/lsa64/"
        )
    if not Path(video_dir).exists():
        raise FileNotFoundError(
            f"LSA64 release directory not found: {video_dir}\n"
            f"LSA64 requires manual download. "
            f"See https://facundoq.github.io/datasets/lsa64/ for instructions."
        )
    mp4_files = list(Path(video_dir).glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(
            f"No .mp4 files found in LSA64 directory: {video_dir}\n"
            f"Ensure the release has been extracted and the correct "
            f"variant directory is specified."
        )
    log.info(
        "LSA64 release directory validated: %s (%d .mp4 files)",
        video_dir, len(mp4_files),
    )
    return {"validated": True, "video_dir": video_dir, "mp4_count": len(mp4_files)}


def load_lsa64_class_map(
    source: LSA64SourceConfig,
    log: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Load class map from source config or bundled asset.

    Returns None when the class map is not found and
    ``allow_missing_class_map`` is True.
    """
    candidates = []
    if source.class_map_file:
        candidates.append(Path(source.class_map_file))
    candidates.append(_BUNDLED_CLASS_MAP)

    for path in candidates:
        if path.exists():
            log.info("Loading LSA64 class map from %s", path)
            return load_class_map(
                path,
                id_column="CLASS_ID",
                gloss_column="GLOSS",
                extra_columns=["HANDEDNESS"],
            )

    if source.allow_missing_class_map:
        log.warning(
            "LSA64 class map not found (searched: %s). "
            "Proceeding without GLOSS/HANDEDNESS labels.",
            [str(c) for c in candidates],
        )
        return None

    raise FileNotFoundError(
        f"LSA64 class map not found. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nProvide a class_map_file in your config or place "
        "lsa64_class_map.tsv in the assets/ directory.\n"
        "Set allow_missing_class_map: true to skip class labels."
    )
