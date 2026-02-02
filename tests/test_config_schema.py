"""Tests for Pydantic configuration models (schema.py)."""

import pytest
from pydantic import ValidationError

from sign_prep.config.schema import (
    ClipVideoConfig,
    Config,
    DownloadConfig,
    ExtractorConfig,
    ManifestConfig,
    NormalizeConfig,
    PathsConfig,
    PipelineConfig,
    ProcessingConfig,
    WebDatasetConfig,
)


class TestConfigMinimal:
    """Config builds with minimal fields."""

    def test_dataset_required(self):
        with pytest.raises(ValidationError):
            Config()

    def test_minimal_config(self):
        cfg = Config(dataset="youtube_asl")
        assert cfg.dataset == "youtube_asl"

    def test_all_sub_configs_have_defaults(self):
        cfg = Config(dataset="test")
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.extractor, ExtractorConfig)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.download, DownloadConfig)
        assert isinstance(cfg.manifest, ManifestConfig)
        assert isinstance(cfg.normalize, NormalizeConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.webdataset, WebDatasetConfig)
        assert isinstance(cfg.clip_video, ClipVideoConfig)


class TestSubConfigDefaults:
    """Each sub-config default values match expected."""

    def test_pipeline_defaults(self):
        p = PipelineConfig()
        assert p.mode == "pose"
        assert p.steps == []
        assert p.start_from is None
        assert p.stop_at is None

    def test_download_defaults(self):
        d = DownloadConfig()
        assert d.video_ids_file == ""
        assert d.languages == ["en"]
        assert d.rate_limit == "5M"
        assert d.concurrent_fragments == 5

    def test_manifest_defaults(self):
        m = ManifestConfig()
        assert m.max_text_length == 300
        assert m.min_duration == 0.2
        assert m.max_duration == 60.0

    def test_extractor_defaults(self):
        e = ExtractorConfig()
        assert e.name == "mediapipe"
        assert e.max_workers == 4

    def test_normalize_defaults(self):
        n = NormalizeConfig()
        assert n.mode == "xy_isotropic_z_minmax"
        assert n.remove_z is False
        assert n.reduction is True
        assert n.mask_frame_level is True
        assert n.mask_landmark_level is False
        assert n.visibility_threshold == 0.3
        assert n.unvisible_value == -999.0

    def test_processing_defaults(self):
        p = ProcessingConfig()
        assert p.max_workers == 4
        assert p.target_fps == 24.0
        assert p.frame_skip == 2
        assert p.skip_existing is True

    def test_paths_defaults(self):
        p = PathsConfig()
        assert p.root == ""
        assert p.videos == ""
        assert p.landmarks == ""

    def test_webdataset_defaults(self):
        w = WebDatasetConfig()
        assert w.max_shard_count == 10000
        assert w.max_shard_size is None

    def test_clip_video_defaults(self):
        c = ClipVideoConfig()
        assert c.codec == "copy"
        assert c.resize is None


class TestPipelineModeValidation:
    """PipelineConfig.mode rejects invalid literals."""

    def test_valid_pose(self):
        p = PipelineConfig(mode="pose")
        assert p.mode == "pose"

    def test_valid_video(self):
        p = PipelineConfig(mode="video")
        assert p.mode == "video"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            PipelineConfig(mode="audio")


class TestNormalizeOptionalFields:
    """NormalizeConfig optional fields."""

    def test_keypoint_indices_none(self):
        n = NormalizeConfig()
        assert n.keypoint_indices is None

    def test_keypoint_indices_set(self):
        n = NormalizeConfig(keypoint_indices=[0, 1, 2])
        assert n.keypoint_indices == [0, 1, 2]


class TestTypeCoercion:
    """Test type coercion through Pydantic."""

    def test_int_field(self):
        m = ManifestConfig(max_text_length=500)
        assert m.max_text_length == 500
        assert isinstance(m.max_text_length, int)

    def test_float_field(self):
        m = ManifestConfig(min_duration=1.5)
        assert m.min_duration == 1.5
        assert isinstance(m.min_duration, float)

    def test_bool_field(self):
        n = NormalizeConfig(reduction=False)
        assert n.reduction is False

    def test_list_field(self):
        d = DownloadConfig(languages=["fr", "de"])
        assert d.languages == ["fr", "de"]
