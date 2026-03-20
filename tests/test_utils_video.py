"""Tests for FPSSampler (video.py)."""

from signdata.utils.video import FPSSampler


class TestFPSSamplerReduceMode:
    def test_30_to_24_fps(self):
        """30→24 fps yields ~80% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, reduce_to=24.0, frame_skip_by=1)
        assert sampler.mode == "reduce"

        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.75 <= ratio <= 0.85, f"Expected ~80%, got {ratio:.2%}"

    def test_target_gte_source_keeps_all(self):
        """target >= source keeps every frame."""
        sampler = FPSSampler(src_fps=24.0, reduce_to=30.0, frame_skip_by=1)
        assert sampler.mode == "reduce"

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_exact_halving(self):
        """30→15 fps yields ~50% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, reduce_to=15.0, frame_skip_by=1)
        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.45 <= ratio <= 0.55, f"Expected ~50%, got {ratio:.2%}"


class TestFPSSamplerSkipMode:
    def test_skip_2_yields_every_other(self):
        """skip=2 yields every other frame."""
        sampler = FPSSampler(src_fps=30.0, reduce_to=None, frame_skip_by=2)
        assert sampler.mode == "skip"

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == 50

    def test_skip_1_yields_every_frame(self):
        """skip=1 yields every frame."""
        sampler = FPSSampler(src_fps=30.0, reduce_to=None, frame_skip_by=1)
        assert sampler.mode == "skip"

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_skip_3(self):
        """skip=3 yields every 3rd frame."""
        sampler = FPSSampler(src_fps=30.0, reduce_to=None, frame_skip_by=3)
        total = 90
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == 30


class TestFPSSamplerModeSelection:
    def test_reduce_to_none_selects_skip(self):
        sampler = FPSSampler(src_fps=30.0, reduce_to=None, frame_skip_by=2)
        assert sampler.mode == "skip"

    def test_reduce_to_float_selects_reduce(self):
        sampler = FPSSampler(src_fps=30.0, reduce_to=24.0, frame_skip_by=2)
        assert sampler.mode == "reduce"

    def test_src_fps_zero_selects_skip(self):
        """Zero source fps falls back to skip mode."""
        sampler = FPSSampler(src_fps=0.0, reduce_to=24.0, frame_skip_by=2)
        assert sampler.mode == "skip"
