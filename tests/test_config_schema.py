"""Tests for Pydantic configuration models (schema.py)."""

import pytest
from pydantic import ValidationError

from sign_prep.config.schema import (
    ClipVideoConfig,
    Config,
    CropVideoConfig,
    DetectPersonConfig,
    ExtractorConfig,
    NormalizeConfig,
    PathsConfig,
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
        assert cfg.recipe == "pose"
        assert cfg.run_name == "default"
        assert cfg.source == {}
        assert cfg.stage_config == {}
        assert isinstance(cfg.extractor, ExtractorConfig)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.normalize, NormalizeConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.webdataset, WebDatasetConfig)
        assert isinstance(cfg.clip_video, ClipVideoConfig)
        assert isinstance(cfg.detect_person, DetectPersonConfig)
        assert isinstance(cfg.crop_video, CropVideoConfig)


class TestRecipeValidation:
    """Config.recipe rejects invalid literals."""

    def test_valid_pose(self):
        cfg = Config(dataset="test", recipe="pose")
        assert cfg.recipe == "pose"

    def test_valid_video(self):
        cfg = Config(dataset="test", recipe="video")
        assert cfg.recipe == "video"

    def test_invalid_recipe(self):
        with pytest.raises(ValidationError):
            Config(dataset="test", recipe="audio")


class TestSubConfigDefaults:
    """Each sub-config default values match expected."""

    def test_extractor_defaults(self):
        e = ExtractorConfig()
        assert e.name == "mediapipe"
        assert e.max_workers == 4

    def test_normalize_defaults(self):
        n = NormalizeConfig()
        assert n.mode == "xy_isotropic_z_minmax"
        assert n.remove_z is False
        assert n.select_keypoints is True
        assert n.mask_empty_frames is True
        assert n.mask_low_confidence is False
        assert n.visibility_threshold == 0.3
        assert n.missing_value == -999.0

    def test_processing_defaults(self):
        p = ProcessingConfig()
        assert p.max_workers == 4
        assert p.target_fps == 24.0
        assert p.frame_skip == 2
        assert p.skip_existing is True
        assert p.signer_policy == "primary_signer"

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

    def test_detect_person_defaults(self):
        d = DetectPersonConfig()
        assert d.enabled is False
        assert d.model == "yolov8n.pt"
        assert d.confidence_threshold == 0.5

    def test_crop_video_defaults(self):
        c = CropVideoConfig()
        assert c.enabled is False
        assert c.padding == 0.25
        assert c.codec == "libx264"


class TestNormalizeOptionalFields:
    """NormalizeConfig optional fields."""

    def test_keypoint_preset_none(self):
        n = NormalizeConfig()
        assert n.keypoint_preset is None

    def test_keypoint_preset_set(self):
        n = NormalizeConfig(keypoint_preset="mediapipe_553_to_85")
        assert n.keypoint_preset == "mediapipe_553_to_85"

    def test_keypoint_indices_none(self):
        n = NormalizeConfig()
        assert n.keypoint_indices is None

    def test_keypoint_indices_set(self):
        n = NormalizeConfig(keypoint_indices=[0, 1, 2])
        assert n.keypoint_indices == [0, 1, 2]

    def test_both_preset_and_indices_coexist(self):
        n = NormalizeConfig(
            keypoint_preset="mediapipe_553_to_85",
            keypoint_indices=[0, 1, 2],
        )
        assert n.keypoint_preset == "mediapipe_553_to_85"
        assert n.keypoint_indices == [0, 1, 2]


class TestTypeCoercion:
    """Test type coercion through Pydantic."""

    def test_int_field(self):
        p = ProcessingConfig(max_workers=8)
        assert p.max_workers == 8
        assert isinstance(p.max_workers, int)

    def test_float_field(self):
        p = ProcessingConfig(min_duration=1.5)
        assert p.min_duration == 1.5
        assert isinstance(p.min_duration, float)

    def test_bool_field(self):
        n = NormalizeConfig(select_keypoints=False)
        assert n.select_keypoints is False

    def test_list_field(self):
        e = ExtractorConfig(name="mediapipe")
        assert e.name == "mediapipe"
