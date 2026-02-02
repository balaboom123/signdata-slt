"""Tests for dataset definitions (youtube_asl.py, how2sign.py)."""

import pytest

from sign_prep.config.schema import Config
from sign_prep.datasets.youtube_asl import YouTubeASLDataset
from sign_prep.datasets.how2sign import How2SignDataset


class TestYouTubeASLValidateConfig:
    def test_valid_config_passes(self):
        cfg = Config(
            dataset="youtube_asl",
            download={"video_ids_file": "assets/ids.txt"},
        )
        # Should not raise
        YouTubeASLDataset.validate_config(cfg)

    def test_missing_video_ids_file_raises(self):
        cfg = Config(
            dataset="youtube_asl",
            download={"video_ids_file": ""},
        )
        with pytest.raises(ValueError, match="video_ids_file"):
            YouTubeASLDataset.validate_config(cfg)

    def test_default_download_raises(self):
        cfg = Config(dataset="youtube_asl")
        with pytest.raises(ValueError, match="video_ids_file"):
            YouTubeASLDataset.validate_config(cfg)


class TestHow2SignValidateConfig:
    def test_valid_config_passes(self):
        cfg = Config(
            dataset="how2sign",
            pipeline={"mode": "pose", "steps": ["extract", "normalize", "webdataset"]},
        )
        # Should not raise
        How2SignDataset.validate_config(cfg)

    def test_download_step_raises(self):
        cfg = Config(
            dataset="how2sign",
            pipeline={"mode": "pose", "steps": ["download", "extract"]},
        )
        with pytest.raises(ValueError, match="download"):
            How2SignDataset.validate_config(cfg)

    def test_manifest_step_raises(self):
        cfg = Config(
            dataset="how2sign",
            pipeline={"mode": "pose", "steps": ["manifest", "extract"]},
        )
        with pytest.raises(ValueError, match="manifest"):
            How2SignDataset.validate_config(cfg)
