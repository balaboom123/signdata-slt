"""Base dataset class."""

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import Config


class BaseDataset(ABC):
    """Abstract base class for dataset definitions."""

    name: str

    @classmethod
    def validate_config(cls, config: "Config") -> None:
        """Validate config for this dataset. Override for custom checks."""
        pass
