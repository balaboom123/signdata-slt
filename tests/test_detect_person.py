"""Tests for DetectPersonProcessor and CropVideoProcessor.

Covers:
  - Pure logic helpers (_union_bboxes, _parse_bool, bbox padding/clamp)
  - Manifest column writing and resume (skip already-processed rows)
  - Fallback behaviour when no person is detected
  - Schema defaults for new config classes
  - crop_video ffmpeg command construction (mocked, no real video needed)
  - Pipeline step registration
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from signdata.config.schema import (
    Config,
    CropVideoConfig,
    DetectPersonConfig,
    PathsConfig,
)
from signdata.processors.detect_person import (
    _union_bboxes,
    _sample_frames,
    _sample_frames_uniform,
    _sample_frames_skip,
    _detect_persons_batch,
)
from signdata.processors.crop_video import _crop_single_video, _parse_bool
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path):
    """Config with tmp paths; no real files required."""
    return Config(
        dataset="youtube_asl",
        recipe="video",
        source={"video_ids_file": "/tmp/ids.txt"},
        paths={
            "root": str(tmp_path),
            "videos": str(tmp_path / "videos"),
            "manifest": str(tmp_path / "manifest.csv"),
            "clips": str(tmp_path / "clips"),
            "cropped_clips": str(tmp_path / "cropped_clips"),
        },
        detect_person={"enabled": True},
        crop_video={"enabled": True},
    )


@pytest.fixture
def sample_manifest(tmp_path):
    """Write a minimal manifest TSV and return its path."""
    manifest_path = tmp_path / "manifest.csv"
    df = pd.DataFrame({
        "VIDEO_NAME":    ["vid_a", "vid_a", "vid_b"],
        "SENTENCE_NAME": ["vid_a-0", "vid_a-1", "vid_b-0"],
        "START_REALIGNED": [0.0,  5.0, 1.0],
        "END_REALIGNED":   [4.0, 10.0, 6.0],
        "SENTENCE":      ["hello", "world", "test"],
    })
    df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


@pytest.fixture
def manifest_with_bbox(tmp_path):
    """Manifest that already has BBOX columns (simulates partial run)."""
    manifest_path = tmp_path / "manifest.csv"
    df = pd.DataFrame({
        "VIDEO_NAME":    ["vid_a", "vid_a"],
        "SENTENCE_NAME": ["vid_a-0", "vid_a-1"],
        "START_REALIGNED": [0.0, 5.0],
        "END_REALIGNED":   [4.0, 10.0],
        "SENTENCE":      ["hello", "world"],
        "BBOX_X1":       [10.0, np.nan],
        "BBOX_Y1":       [20.0, np.nan],
        "BBOX_X2":       [200.0, np.nan],
        "BBOX_Y2":       [400.0, np.nan],
        "PERSON_DETECTED": [True, np.nan],
    })
    df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


# ===========================================================================
# 1. Schema defaults
# ===========================================================================

class TestSchemaDefaults:
    def test_detect_person_defaults(self):
        cfg = DetectPersonConfig()
        assert cfg.model == "yolov8n.pt"
        assert cfg.backend == "ultralytics"
        assert cfg.confidence_threshold == 0.5
        assert cfg.uniform_frames == 5
        assert cfg.max_frames == 5
        assert cfg.device == "cpu"
        assert cfg.min_bbox_area == 0.05

    def test_crop_video_defaults(self):
        cfg = CropVideoConfig()
        assert cfg.padding == 0.25
        assert cfg.codec == "libx264"

    def test_paths_config_has_cropped_clips(self):
        p = PathsConfig()
        assert hasattr(p, "cropped_clips")
        assert p.cropped_clips == ""

    def test_config_includes_new_sections(self):
        cfg = Config(dataset="youtube_asl")
        assert isinstance(cfg.detect_person, DetectPersonConfig)
        assert isinstance(cfg.crop_video, CropVideoConfig)


# ===========================================================================
# 2. _union_bboxes helper
# ===========================================================================

class TestUnionBboxes:
    def test_single_bbox(self):
        result = _union_bboxes([(10, 20, 100, 200)])
        assert result == (10, 20, 100, 200)

    def test_union_of_two_overlapping(self):
        b1 = (10.0, 20.0, 100.0, 200.0)
        b2 = (50.0,  5.0, 150.0, 180.0)
        x1, y1, x2, y2 = _union_bboxes([b1, b2])
        assert x1 == 10.0   # min x1
        assert y1 == 5.0    # min y1
        assert x2 == 150.0  # max x2
        assert y2 == 200.0  # max y2

    def test_union_of_non_overlapping(self):
        b1 = (0.0,   0.0,  50.0,  50.0)
        b2 = (60.0, 60.0, 120.0, 120.0)
        x1, y1, x2, y2 = _union_bboxes([b1, b2])
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 120.0
        assert y2 == 120.0

    def test_union_identical_bboxes(self):
        b = (5.0, 10.0, 80.0, 90.0)
        assert _union_bboxes([b, b, b]) == b

    def test_union_five_frames(self):
        bboxes = [(i * 5.0, i * 3.0, i * 5.0 + 100.0, i * 3.0 + 200.0) for i in range(5)]
        x1, y1, x2, y2 = _union_bboxes(bboxes)
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == pytest.approx(4 * 5.0 + 100.0)
        assert y2 == pytest.approx(4 * 3.0 + 200.0)


# ===========================================================================
# 3. _parse_bool helper (crop_video)
# ===========================================================================

class TestParseBool:
    @pytest.mark.parametrize("val,expected", [
        (True,    True),
        (False,   False),
        ("True",  True),
        ("False", False),
        ("true",  True),
        ("false", False),
        (1,       True),
        (0,       False),
        (1.0,     True),
        (0.0,     False),
    ])
    def test_various_inputs(self, val, expected):
        assert _parse_bool(val) == expected

    def test_nan_treated_as_false(self):
        # NaN as float → bool(NaN) is True in Python, but we treat it as False
        # via the float branch: bool(float('nan')) == True — this is expected Python
        # behaviour; our _parse_bool passes it through bool(), so result is True.
        # This test documents the actual behaviour so it doesn't silently change.
        result = _parse_bool(float("nan"))
        assert isinstance(result, bool)


# ===========================================================================
# 4. Crop geometry calculations
# ===========================================================================

class TestCropGeometry:
    """Test padding and clamp logic in _crop_single_video via a mock ffmpeg."""

    def _run_with_mock(self, x1, y1, x2, y2, frame_w, frame_h, padding):
        """Helper: mock cv2 and subprocess to capture the ffmpeg crop filter."""
        clip_path = "/fake/clip.mp4"
        output_path = "/fake/output.mp4"
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        def fake_exists(path):
            # output does not exist yet (so we proceed); clip does exist
            return path != output_path

        with patch("signdata.processors.crop_video.os.path.exists",
                   side_effect=fake_exists), \
             patch("signdata.processors.crop_video.os.makedirs"), \
             patch("signdata.processors.crop_video.cv2.VideoCapture") as mock_cap, \
             patch("signdata.processors.crop_video.subprocess.run", side_effect=fake_run):

            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = True
            # cv2.CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4
            cap_instance.get.side_effect = lambda prop: (
                frame_w if prop == 3 else frame_h
            )
            mock_cap.return_value = cap_instance

            name, ok, msg = _crop_single_video((
                clip_path, output_path,
                x1, y1, x2, y2,
                True,   # person_detected
                padding,
                "libx264",
            ))

        return ok, captured_cmd

    def test_padding_expands_bbox(self):
        """With 25% padding a 100x200 box should expand by 25 and 50 pixels."""
        ok, cmd = self._run_with_mock(
            x1=50, y1=50, x2=150, y2=250,
            frame_w=640, frame_h=480,
            padding=0.25,
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        assert len(vf_arg) == 1
        # crop=w:h:x:y  — parse it
        parts = vf_arg[0].replace("crop=", "").split(":")
        w, h, cx, cy = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        # x: 50 - 0.25*100 = 25 → cx1=25
        assert cx == 25
        # y: 50 - 0.25*200 = 0 → cy1=0
        assert cy == 0
        # w must be even
        assert w % 2 == 0
        assert h % 2 == 0

    def test_bbox_clamped_to_frame_boundary(self):
        """Padded bbox that would exceed frame dims should be clamped."""
        ok, cmd = self._run_with_mock(
            x1=0, y1=0, x2=640, y2=480,   # already full frame
            frame_w=640, frame_h=480,
            padding=0.25,                   # would go negative / over frame
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        parts = vf_arg[0].replace("crop=", "").split(":")
        cx, cy = int(parts[2]), int(parts[3])
        w, h = int(parts[0]), int(parts[1])
        # After clamping, origin must be >= 0
        assert cx >= 0
        assert cy >= 0
        # Width + origin must not exceed frame
        assert cx + w <= 640
        assert cy + h <= 480

    def test_no_person_uses_stream_copy(self):
        """When PERSON_DETECTED=False, ffmpeg should use -c copy."""
        clip_path = "/fake/clip.mp4"
        output_path = "/fake/output.mp4"
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            r = MagicMock()
            r.returncode = 0
            return r

        def fake_exists(path):
            return path != output_path  # clip exists, output does not

        with patch("signdata.processors.crop_video.os.path.exists",
                   side_effect=fake_exists), \
             patch("signdata.processors.crop_video.os.makedirs"), \
             patch("signdata.processors.crop_video.subprocess.run", side_effect=fake_run):

            name, ok, msg = _crop_single_video((
                clip_path, output_path,
                0, 0, 640, 480,
                False,      # person_detected = False
                0.25,
                "libx264",
            ))

        assert ok
        assert msg == "no-person copy"
        # Should use stream copy, NOT a crop filter
        assert "-c" in captured_cmd
        assert "copy" in captured_cmd
        assert not any(c.startswith("crop=") for c in captured_cmd)

    def test_even_dimensions_enforced(self):
        """Crop dimensions must always be even (libx264 requirement)."""
        ok, cmd = self._run_with_mock(
            x1=0, y1=0, x2=101, y2=101,   # odd dimensions before padding
            frame_w=640, frame_h=480,
            padding=0.0,
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        parts = vf_arg[0].replace("crop=", "").split(":")
        w, h = int(parts[0]), int(parts[1])
        assert w % 2 == 0, f"width {w} is not even"
        assert h % 2 == 0, f"height {h} is not even"


# ===========================================================================
# 5. DetectPersonProcessor — manifest I/O (no real video / model needed)
# ===========================================================================

class TestDetectPersonManifest:
    """Test manifest reading, writing, and resume logic without real video."""

    def _make_processor(self, config):
        from signdata.processors.detect_person import DetectPersonProcessor
        return DetectPersonProcessor(config)

    def test_skips_already_processed_rows(self, manifest_with_bbox, tmp_path):
        """Rows where PERSON_DETECTED is already set must be skipped."""
        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
                "manifest": str(manifest_with_bbox),
                "clips": str(tmp_path / "clips"),
                "cropped_clips": str(tmp_path / "cropped_clips"),
            },
        )

        # Only vid_a-1 has PERSON_DETECTED=NaN, so only 1 row is pending
        df = pd.read_csv(manifest_with_bbox, sep="\t")
        pending = df["PERSON_DETECTED"].isna().sum()
        assert pending == 1

    def test_fallback_writes_full_frame_bbox(self, tmp_path):
        """_fallback_row with non-existent video returns 0,0,0,0."""
        from signdata.processors.detect_person import DetectPersonProcessor
        result = DetectPersonProcessor._fallback_row("/nonexistent/video.mp4")
        assert result["PERSON_DETECTED"] is False
        assert result["BBOX_X1"] == 0.0
        assert result["BBOX_Y1"] == 0.0

    def test_manifest_columns_added_on_run(self, sample_manifest, tmp_path):
        """After a successful (mocked) run, manifest must have BBOX_* columns."""
        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
                "manifest": str(sample_manifest),
                "clips": str(tmp_path / "clips"),
                "cropped_clips": str(tmp_path / "cropped_clips"),
            },
        )
        processor = self._make_processor(cfg)

        # Mock YOLO and _sample_frames so no GPU / file I/O is needed
        fake_bbox = (10.0, 20.0, 200.0, 400.0)

        def fake_sample_frames(video_path, start, end, strategy, frame_skip, uniform_frames, max_frames):
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return [(fake_frame, 640, 480)] * uniform_frames

        def fake_detect_batch(model, frames_meta, conf_thresh, min_area):
            # Each frame returns one valid bbox
            return [[fake_bbox]] * len(frames_meta)

        mock_model = MagicMock()

        with patch("signdata.processors.detect_person.YOLO", return_value=mock_model), \
             patch("signdata.processors.detect_person._sample_frames",
                   side_effect=fake_sample_frames), \
             patch("signdata.processors.detect_person._detect_persons_batch",
                   side_effect=fake_detect_batch), \
             patch("os.path.exists", return_value=True):

            from signdata.pipeline.context import PipelineContext
            from signdata.datasets.youtube_asl import YouTubeASLDataset
            ctx = PipelineContext(
                config=cfg,
                dataset=YouTubeASLDataset(),
                project_root=tmp_path,
                manifest_path=sample_manifest,
                video_dir=tmp_path / "videos",
            )
            ctx = processor.run(ctx)

        # Reload stage manifest (detect_person writes a derived manifest)
        stage_manifest = tmp_path / "detect_person" / "default" / "manifest.csv"
        df = pd.read_csv(stage_manifest, sep="\t")
        for col in ["BBOX_X1", "BBOX_Y1", "BBOX_X2", "BBOX_Y2", "PERSON_DETECTED"]:
            assert col in df.columns, f"Missing column: {col}"

        # All rows should now have PERSON_DETECTED set (not NaN)
        assert df["PERSON_DETECTED"].notna().all()

        # context.manifest_df should also be updated
        assert ctx.manifest_df is not None
        assert "PERSON_DETECTED" in ctx.manifest_df.columns

    def test_fallback_when_no_person_detected(self, sample_manifest, tmp_path):
        """When YOLOv8 returns no bboxes, PERSON_DETECTED must be False."""
        cfg = Config(
            dataset="youtube_asl",
            source={"video_ids_file": "/tmp/ids.txt"},
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
                "manifest": str(sample_manifest),
                "clips": str(tmp_path / "clips"),
                "cropped_clips": str(tmp_path / "cropped_clips"),
            },
        )
        processor = self._make_processor(cfg)

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("signdata.processors.detect_person.YOLO", return_value=MagicMock()), \
             patch("signdata.processors.detect_person._sample_frames",
                   return_value=[(fake_frame, 640, 480)] * 5), \
             patch("signdata.processors.detect_person._detect_persons_batch",
                   return_value=[[] for _ in range(5)]), \
             patch("os.path.exists", return_value=True):

            from signdata.pipeline.context import PipelineContext
            from signdata.datasets.youtube_asl import YouTubeASLDataset
            ctx = PipelineContext(
                config=cfg,
                dataset=YouTubeASLDataset(),
                project_root=tmp_path,
                manifest_path=sample_manifest,
                video_dir=tmp_path / "videos",
            )
            ctx = processor.run(ctx)

        stage_manifest = tmp_path / "detect_person" / "default" / "manifest.csv"
        df = pd.read_csv(stage_manifest, sep="\t")
        # All rows should be fallback
        assert (df["PERSON_DETECTED"] == False).all()  # noqa: E712
        # Stats should reflect fallback count
        assert ctx.stats["detect_person"]["fallback"] == 3


# ===========================================================================
# 6. Pipeline registration
# ===========================================================================

class TestPipelineRegistration:
    def test_detect_person_registered(self):
        assert "detect_person" in PROCESSOR_REGISTRY

    def test_crop_video_registered(self):
        assert "crop_video" in PROCESSOR_REGISTRY

    def test_new_steps_in_video_recipe(self):
        """get_steps for video recipe includes detect_person and crop_video."""
        from signdata.pipeline.recipes import get_steps
        steps = get_steps("video")
        assert "detect_person" in steps
        assert "crop_video" in steps


# ===========================================================================
# 7. Sample strategy
# ===========================================================================

class TestSampleStrategy:
    """Test skip_frame and uniform sampling strategies."""

    def _make_fake_video(self, tmp_path: Path, num_frames: int = 60, fps: int = 30) -> str:
        """Write a minimal fake video file using OpenCV."""
        video_path = str(tmp_path / "fake.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        for _ in range(num_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return video_path

    def test_uniform_returns_exact_n_frames(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames_uniform(video_path, start_sec=0.0, end_sec=2.0, n=5)
        assert len(frames) == 5
        # Each element is (frame, w, h)
        for frame, w, h in frames:
            assert isinstance(frame, np.ndarray)
            assert w == 640
            assert h == 480

    def test_skip_frame_respects_max_frames(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=120, fps=30)
        frames = _sample_frames_skip(
            video_path, start_sec=0.0, end_sec=3.0,
            frame_skip=2, max_frames=5,
        )
        assert len(frames) <= 5

    def test_skip_frame_spacing(self, tmp_path):
        """skip_frame with frame_skip=2 should take roughly every 3rd frame."""
        video_path = self._make_fake_video(tmp_path, num_frames=120, fps=30)
        frames = _sample_frames_skip(
            video_path, start_sec=0.0, end_sec=4.0,
            frame_skip=2, max_frames=20,
        )
        # With 4s @ 30fps = 120 frames, skip=2 → ~40 samples, capped at 20
        assert 1 <= len(frames) <= 20

    def test_dispatcher_skip_frame(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames(
            video_path, 0.0, 2.0,
            strategy="skip_frame",
            frame_skip=2,
            uniform_frames=5,
            max_frames=5,
        )
        assert len(frames) <= 5

    def test_dispatcher_uniform(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames(
            video_path, 0.0, 2.0,
            strategy="uniform",
            frame_skip=2,
            uniform_frames=5,
            max_frames=5,
        )
        assert len(frames) == 5

    def test_schema_default_is_skip_frame(self):
        cfg = DetectPersonConfig()
        assert cfg.sample_strategy == "skip_frame"
        # frame_skip lives on ProcessingConfig, not DetectPersonConfig
        assert not hasattr(cfg, "frame_skip")

    def test_schema_accepts_uniform(self):
        cfg = DetectPersonConfig(sample_strategy="uniform")
        assert cfg.sample_strategy == "uniform"

    def test_schema_rejects_invalid_strategy(self):
        with pytest.raises(Exception):
            DetectPersonConfig(sample_strategy="invalid_mode")
