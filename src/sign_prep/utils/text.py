"""Text normalization utilities."""

import re
from typing import Optional, Dict

import ftfy


def normalize_text(
    text: str,
    *,
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    strip_punctuation: bool = False,
) -> str:
    """Normalize text with configurable processing steps.

    Default behavior (no options) matches the original: fix mojibake via ftfy
    and collapse whitespace.  Adapters may enable additional steps via their
    ``text_processing`` source config.

    Parameters
    ----------
    text : str
        Raw text to normalize.
    fix_encoding : bool
        Fix mojibake / encoding errors via ftfy (default True).
    normalize_whitespace : bool
        Replace newlines and collapse runs of whitespace (default True).
    lowercase : bool
        Convert to lowercase (default False).
    strip_punctuation : bool
        Remove punctuation characters (default False).
    """
    if fix_encoding:
        text = ftfy.fix_text(text)

    if normalize_whitespace:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

    if lowercase:
        text = text.lower()

    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse any whitespace left after punctuation removal
        if normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

    return text
