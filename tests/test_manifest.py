"""Tests for manifest building (YouTubeASLDataset._process_segments).

Tests the segment processing logic that was moved from
ManifestProcessor to the YouTubeASLDataset adapter in Phase 2.
"""

import signdata.datasets  # noqa: F401 – trigger registrations
import signdata.processors  # noqa: F401

from signdata.datasets.youtube_asl import YouTubeASLDataset


def _make_adapter():
    return YouTubeASLDataset()


class TestProcessSegments:
    def test_valid_entries_pass_through(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "Test sentence", "start": 3.0, "duration": 1.5},
        ]
        segments = adapter._process_segments(
            transcripts, "vid001",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 2
        assert segments[0]["VIDEO_ID"] == "vid001"
        assert segments[0]["TEXT"] == "Hello world"
        assert segments[0]["START"] == 0.0
        assert segments[0]["END"] == 2.0

    def test_max_text_length_filter(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Short", "start": 0.0, "duration": 1.0},
            {"text": "A" * 301, "start": 1.0, "duration": 1.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid002",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["TEXT"] == "Short"

    def test_min_duration_filter(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Too short", "start": 0.0, "duration": 0.1},
            {"text": "OK length", "start": 1.0, "duration": 0.5},
        ]
        segments = adapter._process_segments(
            transcripts, "vid003",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["TEXT"] == "OK length"

    def test_max_duration_filter(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Too long", "start": 0.0, "duration": 100.0},
            {"text": "Normal", "start": 1.0, "duration": 5.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid004",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["TEXT"] == "Normal"

    def test_missing_required_fields_skipped(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Valid", "start": 0.0, "duration": 1.0},
            {"start": 0.0, "duration": 1.0},            # missing text
            {"text": "No start", "duration": 1.0},       # missing start
            {"text": "No dur", "start": 0.0},             # missing duration
        ]
        segments = adapter._process_segments(
            transcripts, "vid005",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["TEXT"] == "Valid"

    def test_empty_text_after_normalization_filtered(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "   ", "start": 0.0, "duration": 1.0},  # whitespace only
            {"text": "OK", "start": 1.0, "duration": 1.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid006",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["TEXT"] == "OK"

    def test_sentence_name_format(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": f"Sentence {i}", "start": float(i), "duration": 1.0}
            for i in range(5)
        ]
        segments = adapter._process_segments(
            transcripts, "ABC123",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 5
        assert segments[0]["SAMPLE_ID"] == "ABC123-000"
        assert segments[1]["SAMPLE_ID"] == "ABC123-001"
        assert segments[4]["SAMPLE_ID"] == "ABC123-004"

    def test_end_realigned_computed(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Test", "start": 5.0, "duration": 3.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid007",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert segments[0]["END"] == 8.0

    def test_text_options_applied(self):
        """Text processing options are passed through to normalize_text."""
        adapter = _make_adapter()
        transcripts = [
            {"text": "Hello World!", "start": 0.0, "duration": 1.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid008",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
            text_options={"lowercase": True},
        )
        assert segments[0]["TEXT"] == "hello world!"

    def test_text_options_strip_punctuation(self):
        adapter = _make_adapter()
        transcripts = [
            {"text": "Hello, world!", "start": 0.0, "duration": 1.0},
        ]
        segments = adapter._process_segments(
            transcripts, "vid009",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
            text_options={"strip_punctuation": True},
        )
        assert segments[0]["TEXT"] == "Hello world"
