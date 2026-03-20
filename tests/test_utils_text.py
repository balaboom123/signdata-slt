"""Tests for text normalization (text.py)."""

from signdata.utils.text import normalize_text


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


class TestNormalizeTextOptions:
    """Tests for configurable text processing options (Phase 2)."""

    def test_lowercase(self):
        assert normalize_text("Hello World", lowercase=True) == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("Hello, world!", strip_punctuation=True) == "Hello world"

    def test_lowercase_and_strip_punctuation(self):
        result = normalize_text("Hello, World!", lowercase=True, strip_punctuation=True)
        assert result == "hello world"

    def test_disable_fix_encoding(self):
        """When fix_encoding=False, ftfy is not applied."""
        # CafÃ© is mojibake for Café — without ftfy it stays broken
        result = normalize_text("CafÃ©", fix_encoding=False)
        assert result == "CafÃ©"

    def test_disable_normalize_whitespace(self):
        """When normalize_whitespace=False, whitespace is preserved."""
        result = normalize_text("hello\n  world", normalize_whitespace=False)
        assert "\n" in result

    def test_all_defaults_match_original(self):
        """Default options produce identical output to original function."""
        text = "  CafÃ©  \n  hello   world  "
        assert normalize_text(text) == "Café hello world"

    def test_strip_punctuation_collapses_whitespace(self):
        """Removing punctuation doesn't leave double spaces."""
        result = normalize_text("a, b, c", strip_punctuation=True)
        assert result == "a b c"
