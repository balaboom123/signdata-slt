"""CSL source config, path resolution, and release validation."""

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .._shared.availability import AvailabilityPolicy

# Split-I: signer-independent boundary
SPLIT_I_TRAIN_SIGNERS = set(range(1, 41))
SPLIT_I_TEST_SIGNERS = set(range(41, 51))

# Split-II: unseen-sentence boundary
SPLIT_II_TRAIN_SENTENCES = set(range(1, 95))
SPLIT_II_TEST_SENTENCES = set(range(95, 101))

DEFAULT_FPS = 30.0
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class CSLSourceConfig(BaseModel):
    """Typed config for CSL adapter."""

    release_dir: str = ""
    variant: str = "continuous_2015"
    protocol: str = "split_i"
    split: str = "all"
    split_spec_file: str = ""
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    rgb_subdir: str = "color"
    corpus_file: str = ""
    dictionary_file: str = ""


def get_source_config(config) -> CSLSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.videos:
        source_dict["release_dir"] = config.paths.videos
    return CSLSourceConfig(**source_dict)


def resolve_release_dir(source: CSLSourceConfig, config) -> Path:
    raw = source.release_dir or (config.paths.videos or "")
    return Path(raw)


def resolve_corpus_file(
    source: CSLSourceConfig,
    release_dir: Path,
    log: logging.Logger,
) -> Optional[Path]:
    if source.corpus_file:
        p = Path(source.corpus_file)
        if p.exists():
            return p
        log.warning("Configured corpus_file not found: %s", source.corpus_file)

    candidates = [
        release_dir / "corpus.txt",
        release_dir / "corpus.tsv",
        release_dir / "sentences.txt",
        release_dir / "sentences.tsv",
        release_dir / "label.txt",
        release_dir / "label.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_video_dir(release_dir: Path, source: CSLSourceConfig) -> Path:
    if source.rgb_subdir:
        candidate = release_dir / source.rgb_subdir
        if candidate.is_dir():
            return candidate
    return release_dir


def validate(source: CSLSourceConfig, config, log: logging.Logger) -> dict:
    """Validate CSL release directory and corpus file."""
    release_dir = resolve_release_dir(source, config)

    if not str(release_dir).strip():
        raise FileNotFoundError(
            "CSL requires a local release directory. "
            "Set dataset.source.release_dir or paths.videos in your config YAML.\n"
            "Download CSL from http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html"
        )
    if not release_dir.exists():
        raise FileNotFoundError(
            f"CSL release directory not found: {release_dir}\n"
            f"CSL requires manual download. "
            f"See http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html"
        )

    corpus_path = resolve_corpus_file(source, release_dir, log)
    if corpus_path is None:
        raise FileNotFoundError(
            f"CSL corpus file not found under {release_dir}.\n"
            f"Tried: corpus.txt, corpus.tsv, sentences.txt, label.txt.\n"
            f"Set dataset.source.corpus_file explicitly in your config YAML."
        )

    log.info(
        "CSL release validated: dir=%s, corpus=%s, variant=%s, protocol=%s",
        release_dir, corpus_path, source.variant, source.protocol,
    )
    return {
        "validated": True,
        "release_dir": str(release_dir),
        "corpus_file": str(corpus_path),
        "variant": source.variant,
        "protocol": source.protocol,
    }
