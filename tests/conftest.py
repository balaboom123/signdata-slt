"""Shared fixtures for sign_prep test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Ensure src/ is on the path so sign_prep can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def project_root():
    """Return path to project root."""
    return PROJECT_ROOT


@pytest.fixture
def sample_config():
    """Return a minimal valid Config object."""
    from sign_prep.config.schema import Config

    return Config(
        dataset="youtube_asl",
        recipe="pose",
        paths={"root": "/tmp/test_dataset"},
        source={"video_ids_file": "assets/ids.txt"},
    )


@pytest.fixture
def sample_yaml_file(tmp_path):
    """Write a minimal YAML config to tmp and return its path.

    The file is placed inside a ``configs/jobs/`` subdirectory so that
    ``load_config`` can compute *project_root* correctly (it strips a
    ``configs/jobs`` parent when present).
    """
    configs_dir = tmp_path / "configs" / "jobs"
    configs_dir.mkdir(parents=True)
    yaml_path = configs_dir / "test_config.yaml"
    data = {
        "dataset": "youtube_asl",
        "recipe": "pose",
        "extractor": {"name": "mediapipe"},
        "source": {"video_ids_file": "assets/ids.txt"},
    }
    yaml_path.write_text(yaml.dump(data))
    return yaml_path


@pytest.fixture
def synthetic_landmarks():
    """Return np.ndarray shape (10, 85, 4) with realistic-ish values.

    Channels are (x, y, z, visibility) where x/y in [0,1], z small, vis ~1.
    """
    rng = np.random.default_rng(42)
    arr = np.zeros((10, 85, 4), dtype=np.float32)
    arr[..., 0] = rng.uniform(0.2, 0.8, size=(10, 85))  # x
    arr[..., 1] = rng.uniform(0.2, 0.8, size=(10, 85))  # y
    arr[..., 2] = rng.uniform(-0.05, 0.05, size=(10, 85))  # z
    arr[..., 3] = rng.uniform(0.7, 1.0, size=(10, 85))  # visibility
    return arr
