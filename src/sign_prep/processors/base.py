"""Base processor class for pipeline steps."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext

from ..config.schema import Config


class BaseProcessor(ABC):
    """Abstract base class for pipeline processing steps.

    Subclasses must set ``name`` and ``config_hash_fields``, and implement
    ``run()``.  Override ``validate_inputs()`` to declare required inputs
    so that ``--from`` / ``--only`` produce clear error messages.
    """

    name: str  # Must match registry key

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"sign_prep.{self.name}")

    @abstractmethod
    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Execute this processing step. Return updated context."""
        pass

    def validate(self, context: "PipelineContext") -> bool:
        """Check prerequisites. Override for custom validation."""
        return True

    def validate_inputs(self, context: "PipelineContext") -> None:
        """Validate that required inputs exist before running.

        Override in subclasses to check for required directories,
        manifest columns, etc.  Raise with a clear message if
        prerequisites are missing.

        Called by the runner before each stage when using ``--from``
        or ``--only``.
        """
        pass
