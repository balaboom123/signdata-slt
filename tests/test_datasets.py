"""Tests for dataset adapters (youtube_asl.py, how2sign.py).

Covers validate_config, get_source_config, acquire, and build_manifest.
"""

import json
import os

import pandas as pd
import pytest

from sign_prep.config.schema import Config
from sign_prep.datasets.youtube_asl import YouTubeASLDataset, YouTubeASLSourceConfig
from sign_prep.datasets.how2sign import How2SignDataset, How2SignSourceConfig
from sign_prep.datasets.base import DatasetAdapter, BaseDataset
from sign_prep.pipeline.context import PipelineContext
from sign_prep.registry import DATASET_REGISTRY


# ── DatasetAdapter ABC ──────────────────────────────────────────────────────

class TestDatasetAdapterABC:
    def test_base_dataset_alias(self):
        """BaseDataset is an alias for DatasetAdapter."""
        assert BaseDataset is DatasetAdapter

    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate DatasetAdapter directly."""
        with pytest.raises(TypeError):
            DatasetAdapter()

    def test_registered_in_registry(self):
        assert "youtube_asl" in DATASET_REGISTRY
        assert "how2sign" in DATASET_REGISTRY

    def test_adapter_has_required_methods(self):
        """Adapters implement acquire, build_manifest, get_source_config."""
        adapter = YouTubeASLDataset()
        assert hasattr(adapter, "acquire")
        assert hasattr(adapter, "build_manifest")
        assert hasattr(adapter, "get_source_config")
        assert hasattr(adapter, "validate_config")


# ── YouTube-ASL validate_config ─────────────────────────────────────────────

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


# ── YouTube-ASL get_source_config ───────────────────────────────────────────

class TestYouTubeASLSourceConfig:
    def test_source_config_from_existing_config(self):
        """get_source_config bridges existing config fields into typed model."""
        cfg = Config(
            dataset="youtube_asl",
            download={
                "video_ids_file": "assets/ids.txt",
                "languages": ["en", "ase"],
                "rate_limit": "10M",
            },
            manifest={
                "max_text_length": 500,
                "min_duration": 0.5,
            },
        )
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert isinstance(source, YouTubeASLSourceConfig)
        assert source.video_ids_file == "assets/ids.txt"
        assert source.languages == ["en", "ase"]
        assert source.rate_limit == "10M"
        assert source.max_text_length == 500
        assert source.min_duration == 0.5
        assert source.max_duration == 60.0  # default

    def test_source_config_defaults(self):
        cfg = Config(dataset="youtube_asl")
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert source.languages == ["en"]
        assert source.concurrent_fragments == 5
        assert source.text_processing.fix_encoding is True
        assert source.text_processing.lowercase is False

    def test_source_config_bridges_text_processing(self):
        """Text processing fields from ManifestConfig flow into source config."""
        cfg = Config(
            dataset="youtube_asl",
            manifest={"lowercase": True, "strip_punctuation": True},
        )
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert source.text_processing.lowercase is True
        assert source.text_processing.strip_punctuation is True
        assert source.text_processing.fix_encoding is True  # default preserved


# ── YouTube-ASL build_manifest ──────────────────────────────────────────────

class TestYouTubeASLBuildManifest:
    def _make_context(self, config):
        adapter = YouTubeASLDataset()
        from pathlib import Path
        return PipelineContext(
            config=config,
            dataset=adapter,
            project_root=Path("/tmp"),
        )

    def test_build_manifest_from_transcripts(self, tmp_path):
        """build_manifest produces manifest from transcript JSON files."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        # Write a sample transcript
        transcript = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "Second sentence", "start": 3.0, "duration": 1.5},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"

        cfg = Config(
            dataset="youtube_asl",
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert context.manifest_df is not None
        assert len(context.manifest_df) == 2
        assert "VIDEO_ID" in context.manifest_df.columns
        assert "SAMPLE_ID" in context.manifest_df.columns
        assert context.stats["manifest"]["segments"] == 2

    def test_build_manifest_no_transcripts(self, tmp_path):
        """build_manifest handles empty transcript directory."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        cfg = Config(
            dataset="youtube_asl",
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.stats["manifest"]["segments"] == 0

    def test_build_manifest_filters_by_duration(self, tmp_path):
        """build_manifest respects min/max duration."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        transcript = [
            {"text": "Too short", "start": 0.0, "duration": 0.05},
            {"text": "OK", "start": 1.0, "duration": 1.0},
            {"text": "Too long", "start": 5.0, "duration": 100.0},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset="youtube_asl",
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.stats["manifest"]["segments"] == 1
        assert context.manifest_df.iloc[0]["TEXT"] == "OK"

    def test_build_manifest_text_processing_wired(self, tmp_path):
        """manifest.lowercase=True flows through to build_manifest output."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        transcript = [
            {"text": "Hello World!", "start": 0.0, "duration": 2.0},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset="youtube_asl",
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
            manifest={"lowercase": True, "strip_punctuation": True},
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["TEXT"] == "hello world"


# ── How2Sign validate_config ───────────────────────────────────────────────

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

    def test_manifest_step_allowed(self):
        """manifest step is now allowed — adapter's build_manifest loads the CSV."""
        cfg = Config(
            dataset="how2sign",
            pipeline={"mode": "pose", "steps": ["manifest", "extract"]},
        )
        # Should not raise
        How2SignDataset.validate_config(cfg)


# ── How2Sign get_source_config ──────────────────────────────────────────────

class TestHow2SignSourceConfig:
    def test_source_config_from_existing_config(self):
        cfg = Config(
            dataset="how2sign",
            paths={"manifest": "/data/how2sign/manifest.csv"},
        )
        adapter = How2SignDataset()
        source = adapter.get_source_config(cfg)

        assert isinstance(source, How2SignSourceConfig)
        assert source.manifest_csv == "/data/how2sign/manifest.csv"
        assert source.split == "all"


# ── How2Sign acquire ────────────────────────────────────────────────────────

class TestHow2SignAcquire:
    def test_acquire_validates_existing_dir(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        cfg = Config(
            dataset="how2sign",
            paths={"videos": str(video_dir)},
        )
        adapter = How2SignDataset()
        from pathlib import Path
        context = PipelineContext(
            config=cfg, dataset=adapter, project_root=Path("/tmp"),
        )

        # Should not raise
        context = adapter.acquire(cfg, context)
        assert context.stats["acquire"]["validated"] is True

    def test_acquire_missing_dir_raises(self, tmp_path):
        cfg = Config(
            dataset="how2sign",
            paths={"videos": str(tmp_path / "nonexistent")},
        )
        adapter = How2SignDataset()
        from pathlib import Path
        context = PipelineContext(
            config=cfg, dataset=adapter, project_root=Path("/tmp"),
        )

        with pytest.raises(FileNotFoundError, match="How2Sign"):
            adapter.acquire(cfg, context)


# ── How2Sign build_manifest ────────────────────────────────────────────────

class TestHow2SignBuildManifest:
    def test_build_manifest_loads_csv(self, tmp_path):
        """build_manifest loads existing TSV and sets context."""
        manifest_path = tmp_path / "manifest.csv"
        # Write with legacy columns — read_manifest normalizes them
        df = pd.DataFrame({
            "VIDEO_NAME": ["vid1", "vid1", "vid2"],
            "SENTENCE_NAME": ["vid1-000", "vid1-001", "vid2-000"],
            "START_REALIGNED": [0.0, 2.0, 0.0],
            "END_REALIGNED": [2.0, 4.0, 3.0],
            "SENTENCE": ["Hello", "World", "Test"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset="how2sign",
            paths={"manifest": str(manifest_path)},
        )
        adapter = How2SignDataset()
        from pathlib import Path
        context = PipelineContext(
            config=cfg, dataset=adapter, project_root=Path("/tmp"),
        )

        context = adapter.build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert len(context.manifest_df) == 3
        assert context.stats["manifest"]["videos"] == 2
        assert context.stats["manifest"]["segments"] == 3

    def test_build_manifest_missing_file_raises(self, tmp_path):
        cfg = Config(
            dataset="how2sign",
            paths={"manifest": str(tmp_path / "nope.csv")},
        )
        adapter = How2SignDataset()
        from pathlib import Path
        context = PipelineContext(
            config=cfg, dataset=adapter, project_root=Path("/tmp"),
        )

        with pytest.raises(FileNotFoundError, match="manifest"):
            adapter.build_manifest(cfg, context)
