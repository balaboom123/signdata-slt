"""Tests for PipelineContext, recipe step resolution, and PipelineRunner."""

from pathlib import Path

import pandas as pd
import pytest

import sign_prep.datasets  # noqa: F401 – trigger registrations
import sign_prep.processors  # noqa: F401

from sign_prep.config.schema import Config
from sign_prep.datasets.youtube_asl import YouTubeASLDataset
from sign_prep.pipeline.context import PipelineContext
from sign_prep.pipeline.recipes import get_steps, should_run_stage, RECIPES, OPTIONAL_STAGES
from sign_prep.pipeline.runner import PipelineRunner
from sign_prep.registry import PROCESSOR_REGISTRY


# ── PipelineContext ─────────────────────────────────────────────────────────

class TestPipelineContext:
    def test_instantiation(self):
        cfg = Config(dataset="youtube_asl")
        ds = YouTubeASLDataset()
        ctx = PipelineContext(
            config=cfg,
            dataset=ds,
            project_root=Path("/tmp"),
        )
        assert ctx.config is cfg
        assert ctx.dataset is ds
        assert ctx.project_root == Path("/tmp")

    def test_defaults_empty(self):
        cfg = Config(dataset="youtube_asl")
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            project_root=Path("/tmp"),
        )
        assert ctx.manifest_path is None
        assert ctx.manifest_df is None
        assert ctx.video_dir is None
        assert ctx.manifest_producer == ""
        assert ctx.video_dir_producer == ""
        assert ctx.completed_steps == []
        assert ctx.stats == {}


# ── get_steps (recipe-driven) ──────────────────────────────────────────────

class TestGetSteps:
    def test_full_pose_recipe(self):
        steps = get_steps("pose")
        assert steps == RECIPES["pose"]

    def test_full_video_recipe(self):
        steps = get_steps("video")
        assert steps == RECIPES["video"]

    def test_start_from_filtering(self):
        steps = get_steps("pose", start_from="extract")
        assert steps == ["extract", "normalize", "webdataset"]

    def test_stop_at_filtering(self):
        steps = get_steps("pose", stop_at="manifest")
        assert steps == ["acquire", "manifest"]

    def test_start_from_and_stop_at_combined(self):
        steps = get_steps("pose", start_from="clip_video", stop_at="extract")
        assert steps == ["clip_video", "crop_video", "extract"]

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError, match="Unknown recipe"):
            get_steps("audio")

    def test_start_from_not_in_recipe_raises(self):
        with pytest.raises(ValueError, match="--from"):
            get_steps("pose", start_from="nonexistent")

    def test_stop_at_not_in_recipe_raises(self):
        with pytest.raises(ValueError, match="--to"):
            get_steps("pose", stop_at="nonexistent")

    def test_single_stage(self):
        steps = get_steps("pose", start_from="extract", stop_at="extract")
        assert steps == ["extract"]


# ── should_run_stage ───────────────────────────────────────────────────────

class TestShouldRunStage:
    def test_non_optional_always_true(self):
        cfg = Config(dataset="test")
        assert should_run_stage("acquire", cfg) is True
        assert should_run_stage("manifest", cfg) is True
        assert should_run_stage("extract", cfg) is True

    def test_detect_person_disabled_by_default(self):
        cfg = Config(dataset="test")
        assert should_run_stage("detect_person", cfg) is False

    def test_detect_person_enabled(self):
        cfg = Config(dataset="test", detect_person={"enabled": True})
        assert should_run_stage("detect_person", cfg) is True

    def test_crop_video_disabled_by_default(self):
        cfg = Config(dataset="test")
        assert should_run_stage("crop_video", cfg) is False

    def test_crop_video_enabled(self):
        cfg = Config(
            dataset="test",
            crop_video={"enabled": True},
        )
        assert should_run_stage("crop_video", cfg) is True


# ── PipelineRunner init ───────────────────────────────────────────────────

class TestPipelineRunnerInit:
    def test_init_stores_config(self):
        cfg = Config(
            dataset="youtube_asl",
            recipe="pose",
            source={"video_ids_file": "/tmp/ids.txt"},
        )
        runner = PipelineRunner(cfg)
        assert runner.config is cfg
        assert isinstance(runner.dataset, YouTubeASLDataset)

    def test_init_force_all(self):
        cfg = Config(dataset="youtube_asl", recipe="pose")
        runner = PipelineRunner(cfg, force_all=True)
        assert runner.force_all is True
        assert runner._forced_stages == set()

    def test_init_force_stage_cascades(self):
        """--force <stage> forces that stage and all downstream."""
        cfg = Config(dataset="youtube_asl", recipe="pose")
        runner = PipelineRunner(cfg, force_stage="extract")
        # extract and everything after it
        assert "extract" in runner._forced_stages
        assert "normalize" in runner._forced_stages
        assert "webdataset" in runner._forced_stages
        # stages before extract are not forced
        assert "manifest" not in runner._forced_stages


# ── should_run_stage: clip_video timing fix (P2b) ────────────────────────

class TestShouldRunStageClipVideoTiming:
    """clip_video requires both START and END on the same row."""

    def test_clip_video_with_valid_timing(self):
        cfg = Config(dataset="test")
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1"],
            "VIDEO_ID": ["v1"],
            "START": [0.0],
            "END": [2.0],
        })
        assert should_run_stage("clip_video", cfg, df) is True

    def test_clip_video_start_only_no_end(self):
        """START column present but END is all NaN → should NOT run."""
        cfg = Config(dataset="test")
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1"],
            "VIDEO_ID": ["v1"],
            "START": [0.0],
            "END": [pd.NA],
        })
        assert should_run_stage("clip_video", cfg, df) is False

    def test_clip_video_split_across_rows(self):
        """START on row 0, END on row 1 but not same row → should NOT run."""
        cfg = Config(dataset="test")
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1", "s2"],
            "VIDEO_ID": ["v1", "v1"],
            "START": [0.0, pd.NA],
            "END": [pd.NA, 2.0],
        })
        assert should_run_stage("clip_video", cfg, df) is False

    def test_clip_video_no_manifest(self):
        """Without manifest, assume timing is available."""
        cfg = Config(dataset="test")
        assert should_run_stage("clip_video", cfg, None) is True


# ── Obfuscate guard (P2c) ─────────────────────────────────────────────────

class TestObfuscateGuard:
    def test_obfuscate_not_registered(self):
        """obfuscate is in the video recipe but has no registered processor."""
        assert "obfuscate" not in PROCESSOR_REGISTRY
        assert "obfuscate" in OPTIONAL_STAGES

    def test_obfuscate_activated_by_stage_config(self):
        """should_run_stage activates obfuscate when in stage_config."""
        cfg = Config(dataset="test", stage_config={"obfuscate": {"method": "blur"}})
        assert should_run_stage("obfuscate", cfg) is True

    def test_obfuscate_not_activated_by_default(self):
        cfg = Config(dataset="test")
        assert should_run_stage("obfuscate", cfg) is False


# ── Runner checkpoint integration (P2a) ───────────────────────────────────

class TestRunnerCheckpoint:
    def test_get_stage_output_dir(self, tmp_path):
        cfg = Config(
            dataset="youtube_asl",
            recipe="pose",
            paths={
                "root": str(tmp_path),
                "clips": str(tmp_path / "clips"),
                "landmarks": str(tmp_path / "landmarks"),
            },
        )
        runner = PipelineRunner(cfg)

        # acquire and manifest have distinct checkpoint dirs
        assert runner._get_stage_output_dir("acquire") == tmp_path / "acquire" / "default"
        assert runner._get_stage_output_dir("manifest") == tmp_path / "manifest" / "default"
        assert runner._get_stage_output_dir("detect_person") == tmp_path / "detect_person" / "default"
        assert runner._get_stage_output_dir("clip_video") == tmp_path / "clips"
        assert runner._get_stage_output_dir("extract") == tmp_path / "landmarks"

    def test_acquire_manifest_distinct_checkpoint_dirs(self, tmp_path):
        """acquire and manifest must NOT share a checkpoint directory."""
        cfg = Config(
            dataset="youtube_asl",
            recipe="pose",
            paths={"root": str(tmp_path)},
        )
        runner = PipelineRunner(cfg)
        acquire_dir = runner._get_stage_output_dir("acquire")
        manifest_dir = runner._get_stage_output_dir("manifest")
        assert acquire_dir != manifest_dir

    def test_is_forced_with_force_all(self):
        cfg = Config(dataset="youtube_asl", recipe="pose")
        runner = PipelineRunner(cfg, force_all=True)
        assert runner._is_forced("extract") is True
        assert runner._is_forced("manifest") is True

    def test_is_forced_with_force_stage(self):
        cfg = Config(dataset="youtube_asl", recipe="pose")
        runner = PipelineRunner(cfg, force_stage="normalize")
        assert runner._is_forced("normalize") is True
        assert runner._is_forced("webdataset") is True
        assert runner._is_forced("extract") is False

    def test_force_stage_validates_against_recipe(self):
        """--force with a stage not in the recipe must raise ValueError."""
        cfg = Config(dataset="youtube_asl", recipe="pose")
        with pytest.raises(ValueError, match="--force"):
            PipelineRunner(cfg, force_stage="nonexistent_stage")

    def test_seed_routing_sets_context_for_from(self, tmp_path):
        """When using --from, context routing should reflect upstream stages."""
        cfg = Config(
            dataset="youtube_asl",
            recipe="pose",
            start_from="extract",
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
                "manifest": str(tmp_path / "manifest.csv"),
                "clips": str(tmp_path / "clips"),
            },
        )
        runner = PipelineRunner(cfg)

        context = PipelineContext(
            config=cfg,
            dataset=runner.dataset,
            project_root=tmp_path,
            manifest_path=Path(cfg.paths.manifest) if cfg.paths.manifest else None,
            video_dir=Path(cfg.paths.videos) if cfg.paths.videos else None,
        )

        steps = get_steps(cfg.recipe, cfg.start_from, cfg.stop_at)
        runner._seed_routing(context, steps)

        # After seeding: manifest sets both producers; clip_video (optional
        # but assumed active when no manifest_df) updates video_dir_producer.
        assert context.manifest_producer == "manifest"
        assert context.video_dir_producer == "clip_video"
        assert context.video_dir == Path(cfg.paths.clips)
