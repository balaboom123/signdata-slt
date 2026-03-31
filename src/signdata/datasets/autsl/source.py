"""AUTSL source config, path resolution, and release validation."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from .._shared.availability import AvailabilityPolicy

# Modality suffix appended to the base sample key in filenames.
MODALITY_SUFFIX: Dict[str, str] = {
    "rgb": "_color",
    "depth": "_depth",
}

# Known filesystem aliases for split directory names.
SPLIT_ALIASES: Dict[str, List[str]] = {
    "val": ["validation"],
    "validation": ["val"],
}


class AUTSLSourceConfig(BaseModel):
    """Typed config for AUTSL adapter."""

    release_dir: str = ""
    variant: str = "challenge_2021"
    split: str = "train"
    modality: str = "rgb"
    availability_policy: AvailabilityPolicy = "fail_fast"
    allow_unlabeled: bool = False
    class_id_file: str = ""
    train_labels_file: str = ""
    val_labels_file: str = ""
    test_labels_file: str = ""


def get_source_config(config) -> AUTSLSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.videos:
        source_dict["release_dir"] = config.paths.videos
    return AUTSLSourceConfig(**source_dict)


def resolve_release_root(source: AUTSLSourceConfig, config) -> Path:
    raw = source.release_dir or (config.paths.videos or "")
    return Path(raw)


def parse_signer_id(sample_key: str) -> str:
    """Extract the numeric signer ID from a sample key such as 'signer0_sample1'."""
    match = re.match(r"signer(\d+)", sample_key)
    return match.group(1) if match else ""


def discover_split_dir(release_root: Path, split: str) -> Path:
    """Return the first existing split directory under *release_root*.

    Raises
    ------
    FileNotFoundError
        When neither the canonical name nor any alias exists.
    """
    candidates = [split] + SPLIT_ALIASES.get(split, [])
    for name in candidates:
        candidate = release_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"AUTSL split directory not found for split='{split}' under "
        f"{release_root}. Tried: {candidates}. "
        f"Ensure the dataset is extracted correctly."
    )


def discover_class_id_file(release_root: Path) -> Optional[Path]:
    """Search *release_root* recursively for a class correspondence CSV."""
    for pattern in ("SignList*.csv", "classId*.csv", "class_id*.csv",
                    "*class*correspondence*.csv", "*sign_list*.csv"):
        matches = list(release_root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def discover_labels_file(release_root: Path, split: str) -> Optional[Path]:
    """Search *release_root* for the labels file for *split*."""
    candidates = [
        release_root / f"{split}_labels.csv",
        release_root / f"{split}" / f"{split}_labels.csv",
        release_root / f"labels" / f"{split}_labels.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_class_id_file(
    source: AUTSLSourceConfig,
    release_root: Path,
) -> Optional[Path]:
    if source.class_id_file:
        return Path(source.class_id_file)
    return discover_class_id_file(release_root)


def resolve_labels_file(
    split: str,
    source: AUTSLSourceConfig,
    release_root: Path,
) -> Optional[Path]:
    explicit = {
        "train": source.train_labels_file,
        "val": source.val_labels_file,
        "test": source.test_labels_file,
    }.get(split, "")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return discover_labels_file(release_root, split)


def validate(
    source: AUTSLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate AUTSL release directory and required files."""
    release_root = resolve_release_root(source, config)

    if not release_root.exists():
        raise FileNotFoundError(
            f"AUTSL release directory not found: {release_root}\n"
            f"Set dataset.source.release_dir (or paths.videos) to the "
            f"extracted challenge root and try again."
        )

    found_split = None
    for split_name in ("train", "val", "validation", "test"):
        if (release_root / split_name).is_dir():
            found_split = split_name
            break
    if found_split is None:
        raise FileNotFoundError(
            f"AUTSL release directory contains no recognised split "
            f"subdirectories (train/val/test) under {release_root}. "
            f"Ensure the dataset archive is fully extracted."
        )

    class_id_path = resolve_class_id_file(source, release_root)
    if class_id_path is None or not class_id_path.exists():
        raise FileNotFoundError(
            f"AUTSL class correspondence file not found. "
            f"Specify dataset.source.class_id_file or ensure a file "
            f"matching 'SignList*.csv' / 'classId*.csv' exists under "
            f"{release_root}."
        )

    log.info(
        "AUTSL release validated: root=%s, split_found=%s, class_file=%s",
        release_root, found_split, class_id_path,
    )
    return {
        "validated": True,
        "release_root": str(release_root),
        "split_found": found_split,
        "class_id_file": str(class_id_path),
    }
