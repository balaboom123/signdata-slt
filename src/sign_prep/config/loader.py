"""YAML config loading with _base inheritance."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schema import Config
from ..registry import DATASET_REGISTRY


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_absolute(path_str: str) -> bool:
    """Check if a path string is absolute, handling both Windows and POSIX styles.

    On Windows, Path("/abs/path").is_absolute() returns False because there
    is no drive letter. This helper treats leading '/' as absolute on all
    platforms so that POSIX-style paths in YAML configs work correctly.
    """
    return Path(path_str).is_absolute() or path_str.startswith("/")


def resolve_paths(config: Config, project_root: Path) -> Config:
    """Resolve relative paths to absolute paths based on project root."""
    paths = config.paths

    if not paths.root:
        paths.root = str(project_root / "dataset" / config.dataset)

    root = Path(paths.root)
    if not _is_absolute(paths.root):
        root = project_root / root
        paths.root = str(root)

    extractor_name = config.extractor.name

    if not paths.videos:
        paths.videos = str(root / "videos")
    elif not _is_absolute(paths.videos):
        paths.videos = str(project_root / paths.videos)

    if not paths.transcripts:
        paths.transcripts = str(root / "transcripts")
    elif not _is_absolute(paths.transcripts):
        paths.transcripts = str(project_root / paths.transcripts)

    if not paths.manifest:
        paths.manifest = str(root / "manifest.csv")
    elif not _is_absolute(paths.manifest):
        paths.manifest = str(project_root / paths.manifest)

    if not paths.landmarks:
        paths.landmarks = str(root / "landmarks" / extractor_name)
    elif not _is_absolute(paths.landmarks):
        paths.landmarks = str(project_root / paths.landmarks)

    if not paths.normalized:
        paths.normalized = str(root / "normalized" / extractor_name)
    elif not _is_absolute(paths.normalized):
        paths.normalized = str(project_root / paths.normalized)

    if not paths.clips:
        paths.clips = str(root / "clips")
    elif not _is_absolute(paths.clips):
        paths.clips = str(project_root / paths.clips)

    if not paths.cropped_clips:
        paths.cropped_clips = str(root / "cropped_clips")
    elif not _is_absolute(paths.cropped_clips):
        paths.cropped_clips = str(project_root / paths.cropped_clips)

    if not paths.webdataset:
        paths.webdataset = str(
            root / "webdataset" / config.pipeline.mode / extractor_name
        )
    elif not _is_absolute(paths.webdataset):
        paths.webdataset = str(project_root / paths.webdataset)

    # Resolve download.video_ids_file relative to project root
    vid_file = config.download.video_ids_file
    if vid_file and not _is_absolute(vid_file):
        config.download.video_ids_file = str(project_root / vid_file)

    # Resolve extractor model paths relative to project root
    for attr in ("pose_model_config", "pose_model_checkpoint",
                 "det_model_config", "det_model_checkpoint"):
        val = getattr(config.extractor, attr)
        if val and not _is_absolute(val):
            setattr(config.extractor, attr, str(project_root / val))

    return config


def load_config(
    yaml_path: str,
    overrides: Optional[List[str]] = None,
) -> Config:
    """Load config from YAML file with _base inheritance.

    Merge order (later overrides earlier):
    1. Pydantic defaults (hardcoded in schema)
    2. _base YAML (if specified via _base key)
    3. Dataset-specific YAML config file
    4. CLI overrides (key=value pairs)
    """
    yaml_path = os.path.abspath(yaml_path)
    project_root = Path(yaml_path).parent.parent
    if project_root.name == "configs":
        project_root = project_root.parent

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Handle _base inheritance
    if "_base" in raw:
        base_rel = raw.pop("_base")
        configs_dir = Path(yaml_path).parent.parent  # e.g. configs/youtube_asl/ -> configs/
        base_path = configs_dir / base_rel
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        raw = deep_merge(base_raw, raw)

    dataset_name = raw.get("dataset")
    if not dataset_name:
        raise ValueError("Config must specify 'dataset' field")

    # Validate pipeline.steps is present
    pipeline_cfg = raw.get("pipeline", {})
    if not pipeline_cfg.get("steps"):
        raise ValueError(
            "Config must specify 'pipeline.steps'. "
            "Steps are no longer inferred from dataset classes."
        )

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override}")
            key, value = override.split("=", 1)
            _set_nested(raw, key, _parse_value(value))

    config = Config(**raw)
    config = resolve_paths(config, project_root)

    # Optionally run dataset-specific validation
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_cls.validate_config(config)

    return config


def _set_nested(d: Dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot-separated key."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none" or value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value