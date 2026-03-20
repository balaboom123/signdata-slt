"""Tests for signdata.utils.manifest — canonical schema and shared I/O."""

import pandas as pd
import pytest

from signdata.utils.manifest import (
    REQUIRED_COLUMNS,
    TIMING_COLUMNS,
    LABEL_COLUMNS,
    SPATIAL_COLUMNS,
    METADATA_COLUMNS,
    ALL_KNOWN_COLUMNS,
    _normalize_columns,
    read_manifest,
    validate_manifest,
    has_timing,
    resolve_video_path,
    get_timing_columns,
)


# -----------------------------------------------------------------------
# Column constants
# -----------------------------------------------------------------------

class TestColumnConstants:
    def test_required_columns(self):
        assert REQUIRED_COLUMNS == {"SAMPLE_ID", "VIDEO_ID"}

    def test_timing_columns(self):
        assert TIMING_COLUMNS == {"START", "END"}

    def test_all_known_is_superset(self):
        assert REQUIRED_COLUMNS < ALL_KNOWN_COLUMNS
        assert TIMING_COLUMNS < ALL_KNOWN_COLUMNS
        assert LABEL_COLUMNS < ALL_KNOWN_COLUMNS
        assert SPATIAL_COLUMNS < ALL_KNOWN_COLUMNS
        assert METADATA_COLUMNS < ALL_KNOWN_COLUMNS


# -----------------------------------------------------------------------
# _normalize_columns
# -----------------------------------------------------------------------

class TestNormalizeColumns:
    def test_renames_legacy_columns(self):
        df = pd.DataFrame({
            "VIDEO_NAME": ["v1"],
            "SENTENCE_NAME": ["s1"],
            "START_REALIGNED": [1.0],
            "END_REALIGNED": [2.0],
            "SENTENCE": ["hello"],
        })
        result = _normalize_columns(df)
        assert "VIDEO_ID" in result.columns
        assert "SAMPLE_ID" in result.columns
        assert "START" in result.columns
        assert "END" in result.columns
        assert "TEXT" in result.columns
        # Old names should be gone
        assert "VIDEO_NAME" not in result.columns
        assert "SENTENCE_NAME" not in result.columns

    def test_does_not_overwrite_existing_canonical(self):
        """If both VIDEO_NAME and VIDEO_ID exist, VIDEO_NAME is NOT renamed."""
        df = pd.DataFrame({
            "VIDEO_NAME": ["old"],
            "VIDEO_ID": ["canonical"],
        })
        result = _normalize_columns(df)
        assert result["VIDEO_ID"].iloc[0] == "canonical"
        # VIDEO_NAME stays because canonical already exists
        assert "VIDEO_NAME" in result.columns

    def test_no_op_for_canonical_columns(self):
        df = pd.DataFrame({"VIDEO_ID": ["v1"], "SAMPLE_ID": ["s1"]})
        result = _normalize_columns(df)
        assert list(result.columns) == ["VIDEO_ID", "SAMPLE_ID"]

    def test_caption_to_text(self):
        df = pd.DataFrame({"CAPTION": ["hi"]})
        result = _normalize_columns(df)
        assert "TEXT" in result.columns
        assert result["TEXT"].iloc[0] == "hi"

    def test_no_duplicate_text_from_sentence_and_caption(self):
        """When both SENTENCE and CAPTION exist, only the first alias wins."""
        df = pd.DataFrame({"SENTENCE": ["a"], "CAPTION": ["b"]})
        result = _normalize_columns(df)
        # Only one should be renamed to TEXT; the other keeps its original name
        text_cols = [c for c in result.columns if c == "TEXT"]
        assert len(text_cols) == 1


# -----------------------------------------------------------------------
# read_manifest
# -----------------------------------------------------------------------

class TestReadManifest:
    def test_reads_tsv_with_canonical_columns(self, tmp_path):
        tsv = tmp_path / "manifest.csv"
        tsv.write_text(
            "SAMPLE_ID\tVIDEO_ID\tSTART\tEND\n"
            "s1\tv1\t0.0\t1.0\n"
            "s2\tv1\t1.0\t2.0\n"
        )
        df = read_manifest(tsv)
        assert len(df) == 2
        assert list(df.columns) == ["SAMPLE_ID", "VIDEO_ID", "START", "END"]

    def test_normalizes_legacy_columns_by_default(self, tmp_path):
        tsv = tmp_path / "manifest.csv"
        tsv.write_text(
            "VIDEO_NAME\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\tSENTENCE\n"
            "v1\ts1\t0.0\t1.0\thello\n"
        )
        df = read_manifest(tsv)
        assert "VIDEO_ID" in df.columns
        assert "SAMPLE_ID" in df.columns
        assert "START" in df.columns
        assert "END" in df.columns
        assert "TEXT" in df.columns

    def test_normalize_columns_false_preserves_legacy(self, tmp_path):
        tsv = tmp_path / "manifest.csv"
        tsv.write_text(
            "VIDEO_NAME\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\n"
            "v1\ts1\t0.0\t1.0\n"
        )
        df = read_manifest(tsv, normalize_columns=False)
        assert "VIDEO_NAME" in df.columns
        assert "VIDEO_ID" not in df.columns

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            read_manifest(tmp_path / "nonexistent.csv")

    def test_accepts_string_path(self, tmp_path):
        tsv = tmp_path / "manifest.csv"
        tsv.write_text("SAMPLE_ID\tVIDEO_ID\ns1\tv1\n")
        df = read_manifest(str(tsv))
        assert len(df) == 1


# -----------------------------------------------------------------------
# validate_manifest
# -----------------------------------------------------------------------

class TestValidateManifest:
    def test_valid_manifest_no_issues(self):
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1", "s2"],
            "VIDEO_ID": ["v1", "v1"],
            "START": [0.0, 1.0],
            "END": [1.0, 2.0],
        })
        assert validate_manifest(df) == []

    def test_missing_required_columns(self):
        df = pd.DataFrame({"FOO": [1]})
        issues = validate_manifest(df)
        assert any("Missing required columns" in i for i in issues)

    def test_missing_one_required_column(self):
        df = pd.DataFrame({"SAMPLE_ID": ["s1"]})
        issues = validate_manifest(df)
        assert any("VIDEO_ID" in i for i in issues)

    def test_timing_column_partial(self):
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1"],
            "VIDEO_ID": ["v1"],
            "START": [0.0],
        })
        issues = validate_manifest(df)
        assert any("END" in i for i in issues)

    def test_duplicate_sample_id(self):
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1", "s1", "s2"],
            "VIDEO_ID": ["v1", "v1", "v2"],
        })
        issues = validate_manifest(df)
        assert any("duplicate SAMPLE_ID" in i for i in issues)

    def test_no_timing_is_valid(self):
        """Manifest without START/END is valid (isolated signs)."""
        df = pd.DataFrame({
            "SAMPLE_ID": ["s1"],
            "VIDEO_ID": ["v1"],
        })
        assert validate_manifest(df) == []


# -----------------------------------------------------------------------
# has_timing
# -----------------------------------------------------------------------

class TestHasTiming:
    def test_true_with_valid_timing(self):
        df = pd.DataFrame({"START": [0.0], "END": [1.0]})
        assert has_timing(df) is True

    def test_false_without_columns(self):
        df = pd.DataFrame({"SAMPLE_ID": ["s1"]})
        assert has_timing(df) is False

    def test_false_with_all_null(self):
        df = pd.DataFrame({"START": [None], "END": [None]})
        assert has_timing(df) is False

    def test_false_missing_end(self):
        df = pd.DataFrame({"START": [0.0]})
        assert has_timing(df) is False

    def test_false_when_split_across_rows(self):
        """START on one row, END on another — no complete interval."""
        df = pd.DataFrame({
            "START": [1.0, None],
            "END": [None, 2.0],
        })
        assert has_timing(df) is False

    def test_true_when_at_least_one_complete_row(self):
        df = pd.DataFrame({
            "START": [1.0, None],
            "END": [2.0, None],
        })
        assert has_timing(df) is True


# -----------------------------------------------------------------------
# resolve_video_path
# -----------------------------------------------------------------------

class TestResolveVideoPath:
    def test_default_uses_video_id(self):
        row = pd.Series({"VIDEO_ID": "abc123"})
        result = resolve_video_path(row, "/data/videos")
        assert str(result) == "/data/videos/abc123.mp4"

    def test_rel_path_takes_priority(self):
        row = pd.Series({
            "VIDEO_ID": "abc123",
            "REL_PATH": "train/00001.mp4",
        })
        result = resolve_video_path(row, "/data/videos")
        assert str(result) == "/data/videos/train/00001.mp4"

    def test_null_rel_path_falls_back(self):
        row = pd.Series({
            "VIDEO_ID": "abc123",
            "REL_PATH": None,
        })
        result = resolve_video_path(row, "/data/videos")
        assert str(result) == "/data/videos/abc123.mp4"

    def test_blank_rel_path_falls_back(self):
        """Empty string REL_PATH should be treated as missing."""
        row = pd.Series({
            "VIDEO_ID": "abc123",
            "REL_PATH": "",
        })
        result = resolve_video_path(row, "/data/videos")
        assert str(result) == "/data/videos/abc123.mp4"

    def test_whitespace_rel_path_falls_back(self):
        row = pd.Series({
            "VIDEO_ID": "abc123",
            "REL_PATH": "   ",
        })
        result = resolve_video_path(row, "/data/videos")
        assert str(result) == "/data/videos/abc123.mp4"


# -----------------------------------------------------------------------
# get_timing_columns
# -----------------------------------------------------------------------

class TestGetTimingColumns:
    def test_canonical_names(self):
        df = pd.DataFrame({"START": [0.0], "END": [1.0]})
        assert get_timing_columns(df) == ("START", "END")

    def test_legacy_names(self):
        df = pd.DataFrame({
            "START_REALIGNED": [0.0],
            "END_REALIGNED": [1.0],
        })
        assert get_timing_columns(df) == ("START_REALIGNED", "END_REALIGNED")

    def test_raises_on_missing(self):
        df = pd.DataFrame({"SAMPLE_ID": ["s1"]})
        with pytest.raises(ValueError, match="No recognized timestamp columns"):
            get_timing_columns(df)

    def test_canonical_preferred_over_legacy(self):
        """If both START and START_REALIGNED exist, canonical wins."""
        df = pd.DataFrame({
            "START": [0.0],
            "END": [1.0],
            "START_REALIGNED": [0.5],
            "END_REALIGNED": [1.5],
        })
        assert get_timing_columns(df) == ("START", "END")
