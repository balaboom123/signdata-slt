"""Tests for text normalization (text.py)."""

from sign_prep.utils.text import normalize_text


class TestNormalizeText:
    def test_whitespace_collapsing(self):
        assert normalize_text("hello   world") == "hello world"

    def test_newline_normalization(self):
        assert normalize_text("hello\nworld") == "hello world"
        assert normalize_text("hello\rworld") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_strip_leading_trailing(self):
        assert normalize_text("  hello  ") == "hello"

    def test_ftfy_mojibake(self):
        """ftfy fixes mojibake like 'CafÃ©' → 'Café'."""
        result = normalize_text("CafÃ©")
        assert result == "Café"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_mixed_whitespace(self):
        assert normalize_text("  a \n b \r c  ") == "a b c"

    def test_already_clean(self):
        assert normalize_text("Hello world") == "Hello world"
