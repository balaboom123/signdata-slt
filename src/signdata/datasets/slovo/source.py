"""SLoVo source config, path resolution, and release validation."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel

from .._shared.availability import AvailabilityPolicy

# Bundled class map shipped with the package (assets/slovo_class_map.tsv)
_BUNDLED_CLASS_MAP = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "assets"
    / "slovo_class_map.tsv"
)

# Columns that must be present in annotations.csv
REQUIRED_COLUMNS = {"attachment_id", "user_id", "text", "train"}

# Columns that may be present and map to canonical passthrough names
OPTIONAL_PASSTHROUGH = {
    "width": "SRC_WIDTH",
    "height": "SRC_HEIGHT",
    "length": "FRAME_COUNT",
}


class SlovoSourceConfig(BaseModel):
    """Typed config for the SLoVo adapter."""

    release_dir: str = ""
    annotations_csv: str = ""
    variant: str = "trimmed"
    split: str = "all"
    availability_policy: AvailabilityPolicy = "fail_fast"
    class_map_file: str = ""
    class_map_mode: str = "bundled"
    include_background: bool = True
    background_labels: List[str] = ["no_event"]


def get_source_config(config) -> SlovoSourceConfig:
    return SlovoSourceConfig(**dict(config.dataset.source))


def resolve_release_dir(source: SlovoSourceConfig, config) -> str:
    return source.release_dir or (config.paths.videos or "")


def resolve_annotations_csv(source: SlovoSourceConfig, video_dir: str) -> str:
    if source.annotations_csv:
        return source.annotations_csv
    if video_dir:
        return str(Path(video_dir) / "annotations.csv")
    return "annotations.csv"


def parse_train_col(val) -> bool:
    """Robustly parse the ``train`` column value to a boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


def validate(source: SlovoSourceConfig, config, log: logging.Logger) -> dict:
    """Validate SLoVo release directory and annotations."""
    video_dir = resolve_release_dir(source, config)
    if not video_dir:
        raise ValueError(
            "SLoVo requires a release directory. "
            "Set dataset.source.release_dir or paths.videos in your config YAML."
        )
    if not Path(video_dir).exists():
        raise FileNotFoundError(
            f"SLoVo release directory not found: {video_dir}\n"
            "SLoVo requires manual download. "
            "See https://github.com/hukenovs/slovo for instructions."
        )

    annotations_csv = resolve_annotations_csv(source, video_dir)
    if not Path(annotations_csv).exists():
        raise FileNotFoundError(
            f"SLoVo annotations CSV not found: {annotations_csv}\n"
            "Expected annotations.csv inside the release directory, or "
            "provide an explicit path via dataset.source.annotations_csv."
        )

    ann = pd.read_csv(annotations_csv)
    row_count = len(ann)

    log.info(
        "SLoVo release directory validated: %s (%d annotation rows)",
        video_dir, row_count,
    )
    return {"validated": True, "variant": source.variant, "rows_found": row_count}


def get_bundled_class_map_path(source: SlovoSourceConfig) -> Path:
    return Path(source.class_map_file) if source.class_map_file else _BUNDLED_CLASS_MAP
