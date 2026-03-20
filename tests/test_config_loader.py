"""Tests for config loading and merging (loader.py)."""

import os
from pathlib import Path

import pytest
import yaml

from signdata.config.loader import (
    _parse_value,
    _set_nested,
    deep_merge,
    load_config,
    resolve_paths,
)
from signdata.config.schema import Config


# ── deep_merge ──────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_nested_override(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 99}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 99, "c": 2}}

    def test_non_overlapping_keys(self):
        base = {"x": 1}
        override = {"y": 2}
        result = deep_merge(base, override)
        assert result == {"x": 1, "y": 2}

    def test_list_replacement_not_append(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        assert base["a"]["b"] == 1

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}


# ── _parse_value ────────────────────────────────────────────────────────────

class TestParseValue:
    def test_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("TRUE") is True

    def test_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_none(self):
        assert _parse_value("none") is None
        assert _parse_value("None") is None
        assert _parse_value("null") is None

    def test_int(self):
        assert _parse_value("42") == 42
        assert isinstance(_parse_value("42"), int)

    def test_float(self):
        assert _parse_value("3.14") == pytest.approx(3.14)
        assert isinstance(_parse_value("3.14"), float)

    def test_string(self):
        assert _parse_value("hello") == "hello"
        assert isinstance(_parse_value("hello"), str)

    def test_negative_int(self):
        assert _parse_value("-5") == -5

    def test_negative_float(self):
        assert _parse_value("-1.5") == pytest.approx(-1.5)


# ── _set_nested ─────────────────────────────────────────────────────────────

class TestSetNested:
    def test_creates_nested_dict(self):
        d = {}
        _set_nested(d, "a.b.c", 1)
        assert d == {"a": {"b": {"c": 1}}}

    def test_overwrites_existing(self):
        d = {"a": {"b": {"c": 10}}}
        _set_nested(d, "a.b.c", 99)
        assert d["a"]["b"]["c"] == 99

    def test_simple_key(self):
        d = {}
        _set_nested(d, "x", 5)
        assert d == {"x": 5}

    def test_partial_existing(self):
        d = {"a": {"x": 1}}
        _set_nested(d, "a.y", 2)
        assert d == {"a": {"x": 1, "y": 2}}


# ── resolve_paths ───────────────────────────────────────────────────────────

class TestResolvePaths:
    def test_empty_paths_get_defaults(self):
        cfg = Config(dataset="youtube_asl")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert cfg.paths.root == str(project_root / "dataset" / "youtube_asl")
        assert cfg.paths.videos == str(
            project_root / "dataset" / "youtube_asl" / "videos"
        )
        assert cfg.paths.transcripts == str(
            project_root / "dataset" / "youtube_asl" / "transcripts"
        )

    def test_relative_paths_resolve(self):
        cfg = Config(
            dataset="test",
            paths={"root": "data/test", "videos": "data/test/vids"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert cfg.paths.root == str(project_root / "data" / "test")
        assert cfg.paths.videos == str(project_root / "data" / "test" / "vids")

    def test_absolute_paths_unchanged(self):
        cfg = Config(
            dataset="test",
            paths={"root": "/abs/root", "videos": "/abs/videos"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        # Use as_posix() to normalise separators across Linux and Windows
        assert Path(cfg.paths.root).as_posix() == "/abs/root"
        assert Path(cfg.paths.videos).as_posix() == "/abs/videos"

    def test_video_ids_file_relative_resolved(self):
        """source.video_ids_file is resolved relative to project root."""
        cfg = Config(
            dataset="test",
            source={"video_ids_file": "assets/ids.txt"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.source["video_ids_file"] == str(project_root / "assets" / "ids.txt")

    def test_video_ids_file_absolute_unchanged(self):
        cfg = Config(
            dataset="test",
            source={"video_ids_file": "/abs/ids.txt"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert Path(cfg.source["video_ids_file"]).as_posix() == "/abs/ids.txt"

    def test_video_ids_file_empty_unchanged(self):
        cfg = Config(dataset="test")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.source.get("video_ids_file", "") == ""

    def test_webdataset_path_includes_recipe_and_extractor(self):
        cfg = Config(
            dataset="test",
            recipe="pose",
            extractor={"name": "mediapipe"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.webdataset == str(
            expected_root / "webdataset" / "pose" / "mediapipe" / "default"
        )

    def test_extractor_model_paths_resolved(self):
        """Extractor model paths are resolved relative to project root."""
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_config": "src/signdata/models/configs/model.py",
                "pose_model_checkpoint": "src/signdata/models/checkpoints/model.pth",
                "det_model_config": "src/signdata/models/configs/det.py",
                "det_model_checkpoint": "src/signdata/models/checkpoints/det.pth",
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.extractor.pose_model_config == str(
            project_root / "src/signdata/models/configs/model.py"
        )
        assert cfg.extractor.det_model_checkpoint == str(
            project_root / "src/signdata/models/checkpoints/det.pth"
        )

    def test_extractor_model_paths_absolute_unchanged(self):
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_config": "/abs/model.py",
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert Path(cfg.extractor.pose_model_config).as_posix() == "/abs/model.py"

    def test_extractor_checkpoint_paths_fall_back_to_legacy_package_dir(self, tmp_path):
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_checkpoint": "src/signdata/models/checkpoints/model.pth",
                "det_model_checkpoint": "src/signdata/models/checkpoints/det.pth",
            },
        )
        legacy_pose = tmp_path / "src" / "sign_prep" / "models" / "checkpoints" / "model.pth"
        legacy_det = tmp_path / "src" / "sign_prep" / "models" / "checkpoints" / "det.pth"
        legacy_pose.parent.mkdir(parents=True, exist_ok=True)
        legacy_pose.touch()
        legacy_det.touch()

        cfg = resolve_paths(cfg, tmp_path)

        assert cfg.extractor.pose_model_checkpoint == str(legacy_pose)
        assert cfg.extractor.det_model_checkpoint == str(legacy_det)

    def test_extractor_checkpoint_paths_fall_back_to_intermediate_package_dir(self, tmp_path):
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_checkpoint": "src/signdata/models/checkpoints/model.pth",
                "det_model_checkpoint": "src/signdata/models/checkpoints/det.pth",
            },
        )
        intermediate_pose = tmp_path / "src" / "sltpipe" / "models" / "checkpoints" / "model.pth"
        intermediate_det = tmp_path / "src" / "sltpipe" / "models" / "checkpoints" / "det.pth"
        intermediate_pose.parent.mkdir(parents=True, exist_ok=True)
        intermediate_pose.touch()
        intermediate_det.touch()

        cfg = resolve_paths(cfg, tmp_path)

        assert cfg.extractor.pose_model_checkpoint == str(intermediate_pose)
        assert cfg.extractor.det_model_checkpoint == str(intermediate_det)

    def test_extractor_model_paths_prefer_configured_package_dir(self, tmp_path):
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_config": "src/sign_prep/models/configs/model.py",
                "pose_model_checkpoint": "src/signdata/models/checkpoints/model.pth",
            },
        )
        new_checkpoint = tmp_path / "src" / "signdata" / "models" / "checkpoints" / "model.pth"
        legacy_config = tmp_path / "src" / "sign_prep" / "models" / "configs" / "model.py"
        new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        legacy_config.parent.mkdir(parents=True, exist_ok=True)
        new_checkpoint.touch()
        legacy_config.touch()

        cfg = resolve_paths(cfg, tmp_path)

        assert cfg.extractor.pose_model_config == str(legacy_config)
        assert cfg.extractor.pose_model_checkpoint == str(new_checkpoint)

    def test_extractor_model_paths_fall_forward_from_intermediate_package_dir(self, tmp_path):
        cfg = Config(
            dataset="test",
            extractor={
                "name": "mmpose",
                "pose_model_config": "src/sltpipe/models/configs/model.py",
            },
        )
        canonical_config = tmp_path / "src" / "signdata" / "models" / "configs" / "model.py"
        canonical_config.parent.mkdir(parents=True, exist_ok=True)
        canonical_config.touch()

        cfg = resolve_paths(cfg, tmp_path)

        assert cfg.extractor.pose_model_config == str(canonical_config)

    def test_cropped_clips_default_resolved(self):
        """cropped_clips defaults to <root>/cropped_clips/<run_name> when not set."""
        cfg = Config(dataset="test")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.cropped_clips == str(expected_root / "cropped_clips" / "default")

    def test_cropped_clips_relative_resolved(self):
        """Relative cropped_clips path is resolved against project root."""
        cfg = Config(
            dataset="test",
            paths={"cropped_clips": "data/cropped"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.paths.cropped_clips == str(project_root / "data" / "cropped")

    def test_cropped_clips_absolute_unchanged(self):
        """Absolute cropped_clips path is left as-is."""
        cfg = Config(
            dataset="test",
            paths={"cropped_clips": "/abs/cropped"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert Path(cfg.paths.cropped_clips).as_posix() == "/abs/cropped"

    def test_run_name_in_landmarks_path(self):
        """landmarks default includes run_name."""
        cfg = Config(dataset="test", run_name="exp1")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.landmarks == str(
            expected_root / "landmarks" / "mediapipe" / "exp1"
        )

    def test_run_name_in_clips_path(self):
        """clips default includes run_name."""
        cfg = Config(dataset="test", run_name="exp1")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.clips == str(expected_root / "clips" / "exp1")


# ── load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    """Load real YAML files from configs/jobs/ directory."""

    def test_load_youtube_asl_pose_mediapipe(self, project_root):
        # Trigger registrations
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "youtube_asl_pose_mediapipe.yaml")
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path)
        assert cfg.dataset == "youtube_asl"
        assert cfg.recipe == "pose"
        assert cfg.extractor.name == "mediapipe"

    def test_load_how2sign_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "how2sign_pose_mediapipe.yaml")
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path)
        assert cfg.dataset == "how2sign"
        assert cfg.recipe == "pose"

    def test_cli_overrides_apply(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "youtube_asl_pose_mediapipe.yaml")
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path, overrides=["processing.max_workers=16"])
        assert cfg.processing.max_workers == 16

    def test_missing_dataset_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump({
            "recipe": "pose",
        }))
        with pytest.raises(ValueError, match="dataset"):
            load_config(str(yaml_path))

    def test_missing_recipe_raises(self, tmp_path):
        """Config without recipe raises ValueError."""
        import signdata.datasets
        import signdata.processors

        configs_dir = tmp_path / "configs" / "jobs"
        configs_dir.mkdir(parents=True)
        yaml_path = configs_dir / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "dataset": "youtube_asl",
            "source": {"video_ids_file": "assets/ids.txt"},
        }))
        with pytest.raises(ValueError, match="recipe"):
            load_config(str(yaml_path))
