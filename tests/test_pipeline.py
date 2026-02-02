"""Tests for PipelineContext and PipelineRunner._build_processor_chain."""

from pathlib import Path

import pytest

import sign_prep.datasets  # noqa: F401 – trigger registrations
import sign_prep.processors  # noqa: F401

from sign_prep.config.schema import Config
from sign_prep.datasets.youtube_asl import YouTubeASLDataset
from sign_prep.pipeline.context import PipelineContext
from sign_prep.pipeline.runner import PipelineRunner


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
        assert ctx.completed_steps == []
        assert ctx.stats == {}


# ── PipelineRunner._build_processor_chain ───────────────────────────────────

def _make_config(steps, start_from=None, stop_at=None):
    """Helper to build a Config with specific pipeline steps."""
    return Config(
        dataset="youtube_asl",
        pipeline={
            "mode": "pose",
            "steps": steps,
            "start_from": start_from,
            "stop_at": stop_at,
        },
    )


class TestBuildProcessorChain:
    def test_full_steps(self):
        steps = ["download", "manifest", "extract", "normalize", "webdataset"]
        cfg = _make_config(steps)
        runner = PipelineRunner(cfg)
        names = [p.name for p in runner.processors]
        assert names == steps

    def test_start_from_filtering(self):
        steps = ["download", "manifest", "extract", "normalize", "webdataset"]
        cfg = _make_config(steps, start_from="extract")
        runner = PipelineRunner(cfg)
        names = [p.name for p in runner.processors]
        assert names == ["extract", "normalize", "webdataset"]

    def test_stop_at_filtering(self):
        steps = ["download", "manifest", "extract", "normalize", "webdataset"]
        cfg = _make_config(steps, stop_at="manifest")
        runner = PipelineRunner(cfg)
        names = [p.name for p in runner.processors]
        assert names == ["download", "manifest"]

    def test_start_from_and_stop_at_combined(self):
        steps = ["download", "manifest", "extract", "normalize", "webdataset"]
        cfg = _make_config(steps, start_from="manifest", stop_at="normalize")
        runner = PipelineRunner(cfg)
        names = [p.name for p in runner.processors]
        assert names == ["manifest", "extract", "normalize"]

    def test_unknown_step_raises(self):
        steps = ["download", "nonexistent_step"]
        cfg = _make_config(steps)
        with pytest.raises(ValueError, match="Unknown processor"):
            PipelineRunner(cfg)

    def test_start_from_not_in_steps_raises(self):
        steps = ["download", "manifest"]
        cfg = _make_config(steps, start_from="extract")
        with pytest.raises(ValueError, match="start_from"):
            PipelineRunner(cfg)

    def test_stop_at_not_in_steps_raises(self):
        steps = ["download", "manifest"]
        cfg = _make_config(steps, stop_at="normalize")
        with pytest.raises(ValueError, match="stop_at"):
            PipelineRunner(cfg)
