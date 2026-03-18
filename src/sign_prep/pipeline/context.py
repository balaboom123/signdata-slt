"""Pipeline context for shared state between processing steps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from ..config.schema import Config
from ..datasets.base import DatasetAdapter


@dataclass
class PipelineContext:
    """Shared state passed between pipeline processors."""

    config: Config
    dataset: DatasetAdapter
    project_root: Path

    # Populated by processors as they run
    manifest_path: Optional[Path] = None
    manifest_df: Optional["pd.DataFrame"] = None

    # Tracking
    completed_steps: List[str] = field(default_factory=list)
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
