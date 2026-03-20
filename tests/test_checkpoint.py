"""Tests for signdata.pipeline.checkpoint — stage markers and hashing."""

import json

import pandas as pd
import pytest

from signdata.pipeline.checkpoint import (
    SUCCESS_FILENAME,
    STAGE_HASH_FIELDS,
    compute_stage_hash,
    compute_manifest_hash,
    compute_upstream_hash,
    success_content_hash,
    write_success,
    read_success,
    check_success,
    _resolve_dotpath,
)


# -----------------------------------------------------------------------
# _resolve_dotpath
# -----------------------------------------------------------------------

class TestResolveDotpath:
    def test_dict_access(self):
        obj = {"a": {"b": {"c": 42}}}
        assert _resolve_dotpath(obj, "a.b.c") == 42

    def test_attr_access(self):
        class Cfg:
            class Sub:
                name = "mediapipe"
            extractor = Sub()
        assert _resolve_dotpath(Cfg(), "extractor.name") == "mediapipe"

    def test_missing_returns_none(self):
        assert _resolve_dotpath({"a": 1}, "b.c") is None

    def test_single_level(self):
        assert _resolve_dotpath({"x": 10}, "x") == 10


# -----------------------------------------------------------------------
# compute_stage_hash
# -----------------------------------------------------------------------

class TestComputeStageHash:
    def test_returns_sha256_prefix(self):
        h = compute_stage_hash({"a": 1}, "unknown_stage", hash_fields=["a"])
        assert h.startswith("sha256:")

    def test_deterministic(self):
        cfg = {"normalize": {"mode": "isotropic_3d"}}
        h1 = compute_stage_hash(cfg, "s", hash_fields=["normalize.mode"])
        h2 = compute_stage_hash(cfg, "s", hash_fields=["normalize.mode"])
        assert h1 == h2

    def test_different_values_different_hash(self):
        cfg_a = {"normalize": {"mode": "isotropic_3d"}}
        cfg_b = {"normalize": {"mode": "xy_isotropic_z_minmax"}}
        h_a = compute_stage_hash(cfg_a, "s", hash_fields=["normalize.mode"])
        h_b = compute_stage_hash(cfg_b, "s", hash_fields=["normalize.mode"])
        assert h_a != h_b

    def test_uses_registry_when_no_hash_fields(self):
        """When hash_fields is None, falls back to STAGE_HASH_FIELDS."""
        cfg = {"clip_video": {"codec": "copy", "resize": None},
               "processing": {"min_duration": 0.2, "max_duration": 60.0}}
        h = compute_stage_hash(cfg, "clip_video")
        assert h.startswith("sha256:")
        # Same config should be deterministic
        assert h == compute_stage_hash(cfg, "clip_video")

    def test_unknown_stage_empty_fields(self):
        """Unknown stage with no registry entry hashes empty dict."""
        h = compute_stage_hash({}, "nonexistent_stage")
        assert h.startswith("sha256:")


# -----------------------------------------------------------------------
# compute_manifest_hash
# -----------------------------------------------------------------------

class TestComputeManifestHash:
    def test_from_file(self, tmp_path):
        f = tmp_path / "manifest.csv"
        f.write_text("SAMPLE_ID\tVIDEO_ID\ns1\tv1\n")
        h = compute_manifest_hash(f)
        assert h.startswith("sha256:")

    def test_from_dataframe(self):
        df = pd.DataFrame({"SAMPLE_ID": ["s1"], "VIDEO_ID": ["v1"]})
        h = compute_manifest_hash(df)
        assert h.startswith("sha256:")

    def test_missing_file(self, tmp_path):
        h = compute_manifest_hash(tmp_path / "nope.csv")
        assert h == "sha256:missing"

    def test_deterministic_from_df(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        assert compute_manifest_hash(df) == compute_manifest_hash(df)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "m1.csv"
        f2 = tmp_path / "m2.csv"
        f1.write_text("A\tB\n1\t2\n")
        f2.write_text("A\tB\n3\t4\n")
        assert compute_manifest_hash(f1) != compute_manifest_hash(f2)


# -----------------------------------------------------------------------
# compute_upstream_hash
# -----------------------------------------------------------------------

class TestSuccessContentHash:
    def test_hashes_existing_marker(self, tmp_path):
        out = tmp_path / "stage"
        write_success(out, "test", "sha256:cfg", "sha256:man", output_count=1)
        h = success_content_hash(out)
        assert h.startswith("sha256:")
        assert h != "sha256:missing"

    def test_missing_returns_sentinel(self, tmp_path):
        assert success_content_hash(tmp_path / "nope") == "sha256:missing"

    def test_changes_when_marker_changes(self, tmp_path):
        out = tmp_path / "stage"
        write_success(out, "test", "sha256:cfg1", "sha256:man1", output_count=1)
        h1 = success_content_hash(out)
        write_success(out, "test", "sha256:cfg2", "sha256:man1", output_count=1)
        h2 = success_content_hash(out)
        assert h1 != h2

    def test_changes_when_upstream_reruns_same_config(self, tmp_path):
        """Even if config is identical, a new timestamp means a new hash."""
        out = tmp_path / "stage"
        write_success(out, "test", "sha256:cfg", "sha256:man", output_count=1)
        h1 = success_content_hash(out)
        # Rewrite with same config but different timestamp
        import time
        time.sleep(0.01)
        write_success(out, "test", "sha256:cfg", "sha256:man", output_count=1)
        h2 = success_content_hash(out)
        assert h1 != h2


class TestComputeUpstreamHash:
    def test_deterministic(self):
        hashes = ["sha256:aaa", "sha256:bbb"]
        assert compute_upstream_hash(hashes) == compute_upstream_hash(hashes)

    def test_order_independent(self):
        """Sorted internally, so order of input doesn't matter."""
        h1 = compute_upstream_hash(["sha256:bbb", "sha256:aaa"])
        h2 = compute_upstream_hash(["sha256:aaa", "sha256:bbb"])
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = compute_upstream_hash(["sha256:aaa"])
        h2 = compute_upstream_hash(["sha256:bbb"])
        assert h1 != h2


# -----------------------------------------------------------------------
# write_success / read_success
# -----------------------------------------------------------------------

class TestWriteReadSuccess:
    def test_roundtrip(self, tmp_path):
        out = tmp_path / "stage_out"
        write_success(
            out, "extract",
            stage_config_hash="sha256:abc",
            manifest_hash="sha256:def",
            upstream_hash="sha256:ghi",
            output_count=10,
            output_sample=["a.npy", "b.npy"],
        )
        marker = read_success(out)
        assert marker is not None
        assert marker["stage"] == "extract"
        assert marker["stage_config_hash"] == "sha256:abc"
        assert marker["manifest_hash"] == "sha256:def"
        assert marker["upstream_success_hash"] == "sha256:ghi"
        assert marker["output_count"] == 10
        assert marker["output_sample"] == ["a.npy", "b.npy"]
        assert "timestamp" in marker

    def test_creates_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        write_success(out, "test", "h1", "h2")
        assert (out / SUCCESS_FILENAME).exists()

    def test_read_nonexistent(self, tmp_path):
        assert read_success(tmp_path / "nope") is None

    def test_read_malformed(self, tmp_path):
        out = tmp_path / "bad"
        out.mkdir()
        (out / SUCCESS_FILENAME).write_text("not json{{{")
        assert read_success(out) is None


# -----------------------------------------------------------------------
# check_success
# -----------------------------------------------------------------------

class TestCheckSuccess:
    def _write_valid(self, output_dir, sample_files=None):
        """Helper: write a valid marker and optionally create sample files."""
        if sample_files is None:
            sample_files = ["out.npy"]
        for f in sample_files:
            (output_dir / f).write_text("data")
        write_success(
            output_dir, "test",
            stage_config_hash="sha256:cfg",
            manifest_hash="sha256:man",
            upstream_hash="sha256:up",
            output_count=len(sample_files),
            output_sample=sample_files,
        )

    def test_valid_check(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out)
        assert check_success(out, "sha256:cfg", "sha256:man", "sha256:up") is True

    def test_fails_on_config_mismatch(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out)
        assert check_success(out, "sha256:CHANGED", "sha256:man", "sha256:up") is False

    def test_fails_on_manifest_mismatch(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out)
        assert check_success(out, "sha256:cfg", "sha256:CHANGED", "sha256:up") is False

    def test_fails_on_upstream_mismatch(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out)
        assert check_success(out, "sha256:cfg", "sha256:man", "sha256:CHANGED") is False

    def test_fails_on_missing_marker(self, tmp_path):
        assert check_success(tmp_path, "h1", "h2") is False

    def test_fails_on_zero_output_count(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        write_success(out, "test", "sha256:cfg", "sha256:man", output_count=0)
        assert check_success(out, "sha256:cfg", "sha256:man") is False

    def test_default_output_count_is_accepted(self, tmp_path):
        """write_success() without explicit output_count should be accepted."""
        out = tmp_path / "out"
        out.mkdir()
        write_success(out, "test", "sha256:cfg", "sha256:man")
        assert check_success(out, "sha256:cfg", "sha256:man") is True

    def test_fails_when_sample_files_deleted(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out, ["a.npy"])
        # Delete the sample file
        (out / "a.npy").unlink()
        assert check_success(out, "sha256:cfg", "sha256:man", "sha256:up") is False

    def test_passes_without_upstream_check(self, tmp_path):
        """When upstream_hash is empty, upstream is not checked."""
        out = tmp_path / "out"
        out.mkdir()
        self._write_valid(out)
        assert check_success(out, "sha256:cfg", "sha256:man") is True


# -----------------------------------------------------------------------
# STAGE_HASH_FIELDS registry
# -----------------------------------------------------------------------

class TestStageHashFields:
    def test_known_stages_registered(self):
        expected = {
            "acquire", "manifest", "detect_person", "clip_video",
            "crop_video", "extract", "normalize", "webdataset",
        }
        assert expected <= set(STAGE_HASH_FIELDS.keys())

    def test_all_values_are_lists(self):
        for stage, fields in STAGE_HASH_FIELDS.items():
            assert isinstance(fields, list), f"{stage} has non-list fields"
            for f in fields:
                assert isinstance(f, str), f"{stage} field {f} is not str"
