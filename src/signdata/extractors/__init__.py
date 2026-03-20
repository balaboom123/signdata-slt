"""Pose estimation backends (MediaPipe, MMPose)."""

import importlib
import pkgutil

from .base import LandmarkExtractor

# Auto-discover and import all extractor modules to trigger @register_extractor.
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    if _module_name != "base":
        importlib.import_module(f".{_module_name}", __package__)

from .mediapipe import MediaPipeExtractor
from .mmpose import MMPoseExtractor, MultiPersonDetected

__all__ = [
    "LandmarkExtractor",
    "MediaPipeExtractor",
    "MMPoseExtractor",
    "MultiPersonDetected",
]
