"""Tests for ManifestProcessor._process_segments."""

import sign_prep.datasets  # noqa: F401 – trigger registrations
import sign_prep.processors  # noqa: F401

from sign_prep.config.schema import Config
from sign_prep.processors.youtube_asl.manifest import ManifestProcessor


def _make_processor():
    cfg = Config(dataset="youtube_asl")
    return ManifestProcessor(cfg)


class TestProcessSegments:
    def test_valid_entries_pass_through(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "Test sentence", "start": 3.0, "duration": 1.5},
        ]
        segments = proc._process_segments(
            transcripts, "vid001",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 2
        assert segments[0]["VIDEO_NAME"] == "vid001"
        assert segments[0]["SENTENCE"] == "Hello world"
        assert segments[0]["START_REALIGNED"] == 0.0
        assert segments[0]["END_REALIGNED"] == 2.0

    def test_max_text_length_filter(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Short", "start": 0.0, "duration": 1.0},
            {"text": "A" * 301, "start": 1.0, "duration": 1.0},
        ]
        segments = proc._process_segments(
            transcripts, "vid002",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["SENTENCE"] == "Short"

    def test_min_duration_filter(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Too short", "start": 0.0, "duration": 0.1},
            {"text": "OK length", "start": 1.0, "duration": 0.5},
        ]
        segments = proc._process_segments(
            transcripts, "vid003",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["SENTENCE"] == "OK length"

    def test_max_duration_filter(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Too long", "start": 0.0, "duration": 100.0},
            {"text": "Normal", "start": 1.0, "duration": 5.0},
        ]
        segments = proc._process_segments(
            transcripts, "vid004",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["SENTENCE"] == "Normal"

    def test_missing_required_fields_skipped(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Valid", "start": 0.0, "duration": 1.0},
            {"start": 0.0, "duration": 1.0},            # missing text
            {"text": "No start", "duration": 1.0},       # missing start
            {"text": "No dur", "start": 0.0},             # missing duration
        ]
        segments = proc._process_segments(
            transcripts, "vid005",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["SENTENCE"] == "Valid"

    def test_empty_text_after_normalization_filtered(self):
        proc = _make_processor()
        transcripts = [
            {"text": "   ", "start": 0.0, "duration": 1.0},  # whitespace only
            {"text": "OK", "start": 1.0, "duration": 1.0},
        ]
        segments = proc._process_segments(
            transcripts, "vid006",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 1
        assert segments[0]["SENTENCE"] == "OK"

    def test_sentence_name_format(self):
        proc = _make_processor()
        transcripts = [
            {"text": f"Sentence {i}", "start": float(i), "duration": 1.0}
            for i in range(5)
        ]
        segments = proc._process_segments(
            transcripts, "ABC123",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert len(segments) == 5
        assert segments[0]["SENTENCE_NAME"] == "ABC123-000"
        assert segments[1]["SENTENCE_NAME"] == "ABC123-001"
        assert segments[4]["SENTENCE_NAME"] == "ABC123-004"

    def test_end_realigned_computed(self):
        proc = _make_processor()
        transcripts = [
            {"text": "Test", "start": 5.0, "duration": 3.0},
        ]
        segments = proc._process_segments(
            transcripts, "vid007",
            max_text_length=300, min_duration=0.2, max_duration=60.0,
        )
        assert segments[0]["END_REALIGNED"] == 8.0
