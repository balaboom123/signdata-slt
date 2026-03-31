"""Shared path resolution helpers for dataset adapters."""

from pathlib import Path


def resolve_dir(primary: str, fallback: str = "") -> Path:
    """Return a Path from *primary*, falling back to *fallback*.

    Useful for resolving ``release_dir`` with a ``paths.videos`` fallback::

        release_dir = resolve_dir(source.release_dir, config.paths.videos or "")
    """
    return Path(primary or fallback or "")
