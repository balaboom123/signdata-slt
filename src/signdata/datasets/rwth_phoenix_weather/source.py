"""RWTH-PHOENIX-Weather source config, path resolution, and preparation."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from .._shared.availability import AvailabilityPolicy
from .._shared.media import materialize_frames_to_video

ALL_SPLITS = ("train", "dev", "test")


class RWTHPhoenixWeatherSourceConfig(BaseModel):
    """Typed source config for the RWTH-PHOENIX-Weather adapter."""

    release_dir: str = ""
    variant: str = "phoenix_2014_t"
    split: str = "all"
    prepare_mode: str = "validate"
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    video_fps: float = 25.0


def get_source_config(config) -> RWTHPhoenixWeatherSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and getattr(config.paths, "videos", ""):
        source_dict["release_dir"] = config.paths.videos
    return RWTHPhoenixWeatherSourceConfig(**source_dict)


def find_corpus_csvs(release_dir: Path, split: str) -> List[Path]:
    """Locate PHOENIX corpus CSV files for *split* under *release_dir*."""
    pattern = f"PHOENIX-2014-T.{split}.corpus.csv"
    return sorted(release_dir.rglob(pattern))


def derive_clip_id(name: str) -> str:
    """Normalise a PHOENIX folder/id field into a safe file stem."""
    name = name.strip()
    name = name.replace("/*", "").rstrip("/")
    if name.startswith("/"):
        name = name.lstrip("/")
    return name.replace("/", "_").replace(" ", "_")


def prepare(
    source: RWTHPhoenixWeatherSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate the PHOENIX release directory and optionally materialise videos."""
    release_dir = Path(source.release_dir)

    if not release_dir.exists():
        raise FileNotFoundError(
            f"PHOENIX release directory not found: {release_dir}\n"
            "RWTH-PHOENIX-Weather requires manual download from "
            "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/"
        )

    log.info("PHOENIX release directory validated: %s", release_dir)

    splits = ALL_SPLITS if source.split == "all" else (source.split,)
    mode = source.prepare_mode

    if mode == "validate":
        return {"validated": True, "mode": mode}

    overwrite = mode == "rematerialize_all"
    video_dir = Path(config.paths.videos) if getattr(config.paths, "videos", "") else release_dir

    total_materialized = 0
    total_errors = 0
    total_validated = 0

    for split in splits:
        csv_paths = find_corpus_csvs(release_dir, split)
        if not csv_paths:
            log.warning(
                "No corpus CSV found for split '%s' under %s — skipping.",
                split, release_dir,
            )
            continue
        for csv_path in csv_paths:
            mat, err, val = _materialise_split(
                csv_path=csv_path,
                split=split,
                release_dir=release_dir,
                video_dir=video_dir,
                fps=source.video_fps,
                overwrite=overwrite,
                log=log,
            )
            total_materialized += mat
            total_errors += err
            total_validated += val

    log.info(
        "PHOENIX prepare complete: %d materialized, %d validated, %d errors.",
        total_materialized, total_validated, total_errors,
    )
    return {
        "mode": mode,
        "validated": total_validated,
        "materialized": total_materialized,
        "errors": total_errors,
    }


def _materialise_split(
    csv_path: Path,
    split: str,
    release_dir: Path,
    video_dir: Path,
    fps: float,
    overwrite: bool,
    log: logging.Logger,
) -> Tuple[int, int, int]:
    """Materialise videos for a single corpus CSV.

    Returns (materialized_count, error_count, validated_count).
    """
    try:
        df = pd.read_csv(csv_path, delimiter="|")
    except Exception as exc:
        log.error("Failed to read corpus CSV %s: %s", csv_path, exc)
        return 0, 1, 0

    id_col = "id" if "id" in df.columns else ("name" if "name" in df.columns else None)
    folder_col = "folder" if "folder" in df.columns else None

    if id_col is None:
        log.error("Corpus CSV %s has no 'id' or 'name' column — skipping.", csv_path)
        return 0, 1, 0

    materialized = 0
    errors = 0
    validated = 0

    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        clip_id = derive_clip_id(raw_id)

        raw_folder = str(row[folder_col]) if folder_col and pd.notna(row[folder_col]) else raw_id
        frame_dir = release_dir / raw_folder.lstrip("/")
        output_path = video_dir / split / f"{clip_id}.mp4"

        if not frame_dir.is_dir():
            log.debug(
                "Frame directory not found, skipping clip '%s': %s",
                clip_id, frame_dir,
            )
            errors += 1
            continue

        try:
            materialize_frames_to_video(frame_dir, output_path, fps=fps, overwrite=overwrite)
            if output_path.exists():
                materialized += 1
            else:
                validated += 1
        except Exception as exc:
            log.warning("Failed to materialise clip '%s': %s", clip_id, exc)
            errors += 1

    return materialized, errors, validated
