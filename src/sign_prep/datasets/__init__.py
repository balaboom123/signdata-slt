"""Dataset definitions with config validation."""

import importlib
import pkgutil

from .base import DatasetAdapter, BaseDataset

# Auto-discover and import all dataset modules to trigger @register_dataset.
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    if _module_name != "base":
        importlib.import_module(f".{_module_name}", __package__)

from .youtube_asl import YouTubeASLDataset
from .how2sign import How2SignDataset

__all__ = [
    "DatasetAdapter",
    "BaseDataset",
    "YouTubeASLDataset",
    "How2SignDataset",
]
