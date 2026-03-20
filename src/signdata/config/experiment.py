"""Experiment config schema and loader.

An experiment is an ordered list of pipeline jobs, each referencing a
job config YAML with optional per-job overrides.  This enables
multi-dataset or multi-extractor research workflows in a single command.

Example experiment YAML::

    name: "YouTube-ASL Baseline Reproduction"
    jobs:
      - config: jobs/youtube_asl_pose_mediapipe.yaml
        overrides:
          processing.target_fps: null
          normalize.mask_landmark_level: true
      - config: jobs/youtube_asl_pose_mmpose.yaml
        overrides:
          extractor.device: "cuda:1"
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class JobEntry(BaseModel):
    """A single pipeline job within an experiment."""

    config: str  # path to job YAML (relative to configs/ or absolute)
    overrides: Dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    """Top-level experiment definition.

    Attributes:
        name: Human-readable experiment name (for logging).
        description: Optional longer description.
        jobs: Ordered list of pipeline jobs to execute.
    """

    name: str
    description: str = ""
    jobs: List[JobEntry]


def _flatten_overrides(
    d: Dict[str, Any], prefix: str = "",
) -> Dict[str, Any]:
    """Flatten a possibly-nested dict into dot-separated key → value pairs.

    Supports both formats in experiment YAML::

        # Flat (dot-separated keys) — preferred
        overrides:
          processing.target_fps: null

        # Nested — also accepted
        overrides:
          processing:
            target_fps: null

    Both produce ``{"processing.target_fps": None}``.
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_overrides(v, key))
        else:
            result[key] = v
    return result


def load_experiment(yaml_path: str) -> ExperimentConfig:
    """Load an experiment config from YAML.

    Job config paths are resolved relative to the nearest ``configs/``
    ancestor directory.  This handles any nesting depth (e.g.
    ``configs/experiments/baselines/foo.yaml``).  If no ``configs/``
    ancestor exists, paths resolve relative to the experiment file's
    directory.

    Args:
        yaml_path: Path to the experiment YAML file.

    Returns:
        Validated :class:`ExperimentConfig` with resolved job paths.
    """
    yaml_path = os.path.abspath(yaml_path)
    experiment_dir = Path(yaml_path).parent

    # Determine configs root for resolving job config paths.
    # Walk up from experiment_dir to find a ``configs/`` ancestor.
    # This handles any nesting depth under configs/:
    #   configs/experiments/foo.yaml              → configs/
    #   configs/experiments/baselines/foo.yaml    → configs/
    #   configs/experiments/a/b/c/foo.yaml        → configs/
    # If no ``configs/`` ancestor exists, fall back to experiment_dir.
    configs_root = experiment_dir
    cursor = experiment_dir
    while cursor != cursor.parent:
        if cursor.name == "configs":
            configs_root = cursor
            break
        cursor = cursor.parent

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "name" not in raw:
        raise ValueError("Experiment config must specify a 'name' field")
    if "jobs" not in raw or not raw["jobs"]:
        raise ValueError("Experiment config must have at least one job")

    experiment = ExperimentConfig(**raw)

    # Resolve relative job config paths
    for job in experiment.jobs:
        if not os.path.isabs(job.config):
            resolved = configs_root / job.config
            job.config = str(resolved)

    return experiment
