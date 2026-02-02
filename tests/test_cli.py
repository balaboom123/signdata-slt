"""Tests for CLI argument parsing (cli.py)."""

import pytest

from sign_prep.cli import parse_args


class TestParseArgs:
    def test_config_only(self):
        args = parse_args(["config.yaml"])
        assert args.config == "config.yaml"
        assert args.override == []

    def test_config_with_overrides(self):
        args = parse_args(["config.yaml", "--override", "a=1", "b=2"])
        assert args.config == "config.yaml"
        assert args.override == ["a=1", "b=2"]

    def test_missing_config_raises(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_override_empty_list(self):
        args = parse_args(["my.yaml", "--override"])
        assert args.override == []

    def test_config_path_preserved(self):
        args = parse_args(["/path/to/configs/test.yaml"])
        assert args.config == "/path/to/configs/test.yaml"
