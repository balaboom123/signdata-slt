"""Pipeline step implementations."""

import importlib
import pkgutil

from .base import BaseProcessor

# Auto-discover and import all processor modules to trigger
# @register_processor decorators.
for _, _module_name, _is_pkg in pkgutil.iter_modules(__path__):
    if _module_name != "base":
        importlib.import_module(f".{_module_name}", __package__)

from .extract import ExtractProcessor
from .normalize import NormalizeProcessor
from .clip_video import ClipVideoProcessor
from .detect_person import DetectPersonProcessor
from .crop_video import CropVideoProcessor
from .webdataset import WebDatasetProcessor

__all__ = [
    "BaseProcessor",
    "ExtractProcessor",
    "NormalizeProcessor",
    "ClipVideoProcessor",
    "DetectPersonProcessor",
    "CropVideoProcessor",
    "WebDatasetProcessor",
]
