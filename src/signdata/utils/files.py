"""File and directory utilities."""

import os
from glob import glob
from typing import List


def get_video_filenames(directory: str, pattern: str = "*.mp4") -> List[str]:
    """Retrieve filenames from directory without extensions."""
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(directory, pattern))
    ]


def get_filenames(directory: str, pattern: str, extension: str) -> List[str]:
    """Retrieve filenames with any pattern/extension, without extensions."""
    search_pattern = f"{pattern}.{extension}"
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(directory, search_pattern))
    ]
