"""Component registry for datasets, processors, and extractors."""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets.base import BaseDataset
    from .processors.base import BaseProcessor
    from .extractors.base import LandmarkExtractor

DATASET_REGISTRY: Dict[str, Type["BaseDataset"]] = {}
PROCESSOR_REGISTRY: Dict[str, Type["BaseProcessor"]] = {}
EXTRACTOR_REGISTRY: Dict[str, Type["LandmarkExtractor"]] = {}


def register_dataset(name: str):
    """Register a dataset class under the given name."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def register_processor(name: str):
    """Register a processor class under the given name."""
    def decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_extractor(name: str):
    """Register an extractor class under the given name."""
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator
