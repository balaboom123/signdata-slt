#!/usr/bin/env python3
"""Convenience wrapper for running the sign_prep pipeline.

To smoke-test person_localize + crop_video on a single video without a full
dataset, use the standalone helper:

    python scripts/smoke_test_localize.py --video path/to/video.mp4
"""

import sys
import os

# Add src/ to path so sign_prep is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sign_prep.__main__ import main

if __name__ == "__main__":
    main()
