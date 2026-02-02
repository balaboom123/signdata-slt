"""Tests for file utilities (files.py)."""

import os

from sign_prep.utils.files import get_filenames, get_video_filenames


class TestGetVideoFilenames:
    def test_returns_basenames_without_extension(self, tmp_path):
        for name in ["abc.mp4", "def.mp4", "ghi.mp4"]:
            (tmp_path / name).touch()

        result = get_video_filenames(str(tmp_path))
        assert sorted(result) == ["abc", "def", "ghi"]

    def test_ignores_non_mp4(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.npy").touch()

        result = get_video_filenames(str(tmp_path))
        assert result == ["video"]

    def test_empty_directory(self, tmp_path):
        result = get_video_filenames(str(tmp_path))
        assert result == []

    def test_custom_pattern(self, tmp_path):
        (tmp_path / "a.avi").touch()
        (tmp_path / "b.avi").touch()

        result = get_video_filenames(str(tmp_path), pattern="*.avi")
        assert sorted(result) == ["a", "b"]


class TestGetFilenames:
    def test_with_pattern_and_extension(self, tmp_path):
        for name in ["seg-001.npy", "seg-002.npy", "other.npy"]:
            (tmp_path / name).touch()

        result = get_filenames(str(tmp_path), "seg-*", "npy")
        assert sorted(result) == ["seg-001", "seg-002"]

    def test_empty_directory(self, tmp_path):
        result = get_filenames(str(tmp_path), "*", "json")
        assert result == []

    def test_wildcard_pattern(self, tmp_path):
        (tmp_path / "x.json").touch()
        (tmp_path / "y.json").touch()

        result = get_filenames(str(tmp_path), "*", "json")
        assert sorted(result) == ["x", "y"]
