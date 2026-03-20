"""Tests for ObfuscateProcessor.

Covers:
  - ObfuscateConfig validation and defaults
  - _obfuscate_single_video worker (mocked cv2 + mediapipe)
  - Processor run() — task building, file naming, stats
  - validate_inputs
  - Registration
"""

import os
import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class _SyncExecutor:
    """Drop-in replacement for ProcessPoolExecutor that runs synchronously.

    Avoids subprocess pickling issues when the worker function is mocked.
    """

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            result = fn(*args, **kwargs)
            f.set_result(result)
        except Exception as e:
            f.set_exception(e)
        return f

from signdata.config.schema import Config
from signdata.processors.obfuscate import (
    ObfuscateConfig,
    ObfuscateProcessor,
    _obfuscate_single_video,
)
from signdata.pipeline.context import PipelineContext
from signdata.datasets.youtube_asl import YouTubeASLDataset
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ===========================================================================
# 1. ObfuscateConfig
# ===========================================================================

class TestObfuscateConfig:
    def test_defaults(self):
        cfg = ObfuscateConfig()
        assert cfg.method == "blur"
        assert cfg.blur_strength == 51
        assert cfg.pixelate_size == 10
        assert cfg.min_detection_confidence == 0.5

    def test_pixelate_method(self):
        cfg = ObfuscateConfig(method="pixelate")
        assert cfg.method == "pixelate"

    def test_even_blur_strength_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            ObfuscateConfig(blur_strength=50)

    def test_zero_blur_strength_rejected(self):
        with pytest.raises(ValueError, match="positive odd"):
            ObfuscateConfig(blur_strength=0)

    def test_valid_odd_blur_strength(self):
        cfg = ObfuscateConfig(blur_strength=31)
        assert cfg.blur_strength == 31

    def test_zero_pixelate_size_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ObfuscateConfig(pixelate_size=0)

    def test_negative_pixelate_size_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ObfuscateConfig(pixelate_size=-5)


# ===========================================================================
# 2. _obfuscate_single_video worker
# ===========================================================================

class TestObfuscateSingleVideo:
    def test_skips_existing_output(self):
        """If output already exists and skip_existing=True, return skip message."""
        with patch("signdata.processors.obfuscate.os.path.exists", return_value=True):
            name, ok, msg = _obfuscate_single_video((
                "/fake/in.mp4", "/fake/out.mp4",
                "blur", 51, 10, 0.5, True,
            ))
        assert ok
        assert msg == "skipped (exists)"

    def test_rebuilds_when_not_skip_existing(self):
        """When skip_existing=False, existing output is rebuilt, not skipped."""
        def fake_exists(path):
            if path == "/fake/out.mp4":
                return True   # output exists
            return False      # input doesn't exist

        with patch("signdata.processors.obfuscate.os.path.exists", side_effect=fake_exists):
            name, ok, msg = _obfuscate_single_video((
                "/fake/in.mp4", "/fake/out.mp4",
                "blur", 51, 10, 0.5, False,
            ))
        # Should NOT skip — proceeds to check input, which is missing
        assert not ok
        assert "not found" in msg

    def test_missing_input_returns_error(self):
        """If input video doesn't exist, return error."""
        def fake_exists(path):
            if path == "/fake/out.mp4":
                return False  # output doesn't exist
            return False  # input doesn't exist either

        with patch("signdata.processors.obfuscate.os.path.exists", side_effect=fake_exists):
            name, ok, msg = _obfuscate_single_video((
                "/fake/in.mp4", "/fake/out.mp4",
                "blur", 51, 10, 0.5, True,
            ))
        assert not ok
        assert "not found" in msg


# ===========================================================================
# 3. Processor run() — task building
# ===========================================================================

class TestObfuscateProcessorRun:
    def _make_context(self, cfg, manifest_path, tmp_path, video_dir_producer="clip_video"):
        return PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            project_root=tmp_path,
            manifest_path=manifest_path,
            video_dir=tmp_path / "clips",
            video_dir_producer=video_dir_producer,
            stage_output_dir=tmp_path / "obfuscated" / "default",
        )

    def test_uses_sample_id_when_clipped(self, tmp_path):
        """After clip_video, files are named SAMPLE_ID.mp4."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_a"],
            "SAMPLE_ID": ["vid_a-0", "vid_a-1"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "vid_a-0.mp4").touch()
        (clips_dir / "vid_a-1.mp4").touch()

        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {"method": "blur"}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)

        # Use sync executor + mock worker to avoid subprocess pickling
        def fake_worker(args):
            return os.path.basename(args[1]), True, "ok"

        with patch(
            "signdata.processors.obfuscate.ProcessPoolExecutor", _SyncExecutor,
        ), patch(
            "signdata.processors.obfuscate._obfuscate_single_video", fake_worker,
        ):
            ctx = processor.run(ctx)

        assert ctx.stats["obfuscate"]["total"] == 2
        assert ctx.stats["obfuscate"]["success"] == 2

    def test_uses_video_id_when_not_clipped(self, tmp_path):
        """Before clip_video, files are named VIDEO_ID.mp4."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_a"],
            "SAMPLE_ID": ["vid_a-0", "vid_a-1"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "vid_a.mp4").touch()

        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {"method": "blur"}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = self._make_context(
            cfg, manifest_path, tmp_path, video_dir_producer="manifest",
        )

        def fake_worker(args):
            return os.path.basename(args[1]), True, "ok"

        with patch(
            "signdata.processors.obfuscate.ProcessPoolExecutor", _SyncExecutor,
        ), patch(
            "signdata.processors.obfuscate._obfuscate_single_video", fake_worker,
        ):
            ctx = processor.run(ctx)

        # Two manifest rows but same VIDEO_ID → deduplicated to 1 task
        assert ctx.stats["obfuscate"]["total"] == 1

    def test_skip_existing_false_on_rerun(self, tmp_path):
        """When _SUCCESS.json exists (re-run), skip_existing should be False."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a-0"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "vid_a-0.mp4").touch()

        # Create _SUCCESS.json in output dir to simulate prior completion
        output_dir = tmp_path / "obfuscated" / "default"
        output_dir.mkdir(parents=True)
        (output_dir / "_SUCCESS.json").write_text('{"stage": "obfuscate"}')

        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {"method": "blur"}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)

        captured_args = []
        def fake_worker(args):
            captured_args.append(args)
            return os.path.basename(args[1]), True, "ok"

        with patch(
            "signdata.processors.obfuscate.ProcessPoolExecutor", _SyncExecutor,
        ), patch(
            "signdata.processors.obfuscate._obfuscate_single_video", fake_worker,
        ):
            ctx = processor.run(ctx)

        # The 7th element (skip_existing) should be False on re-run
        assert len(captured_args) == 1
        assert captured_args[0][6] is False

    def test_no_tasks_when_no_videos(self, tmp_path):
        """If no video files exist, stats show total=0."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a-0"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        # No actual video files

        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {"method": "blur"}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)
        ctx = processor.run(ctx)

        assert ctx.stats["obfuscate"]["total"] == 0


# ===========================================================================
# 4. validate_inputs
# ===========================================================================

class TestObfuscateValidateInputs:
    def test_raises_when_no_video_dir(self, tmp_path):
        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            project_root=tmp_path,
            manifest_path=tmp_path / "manifest.csv",
            video_dir=None,
        )
        with pytest.raises(RuntimeError, match="video directory"):
            processor.validate_inputs(ctx)

    def test_raises_when_manifest_missing(self, tmp_path):
        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={"root": str(tmp_path)},
            stage_config={"obfuscate": {}},
        )
        processor = ObfuscateProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            project_root=tmp_path,
            manifest_path=tmp_path / "nonexistent.csv",
            video_dir=tmp_path / "videos",
        )
        with pytest.raises(RuntimeError, match="manifest not found"):
            processor.validate_inputs(ctx)


# ===========================================================================
# 5. Registration
# ===========================================================================

class TestObfuscateRegistration:
    def test_registered(self):
        assert "obfuscate" in PROCESSOR_REGISTRY
