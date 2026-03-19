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
    """Shared state passed between pipeline processors.

    The runner maintains ``manifest_path`` / ``video_dir`` and their
    corresponding ``*_producer`` fields in lockstep after each stage.
    Processors read from these routing fields instead of hardcoding
    paths from config.
    """

    config: Config
    dataset: DatasetAdapter
    project_root: Path

    # Artifact routing — set by the runner after each stage
    manifest_path: Optional[Path] = None
    manifest_df: Optional["pd.DataFrame"] = None
    video_dir: Optional[Path] = None

    # Set by the runner before each stage — processors that write
    # derived artifacts (e.g. detect_person stage manifest) read this
    # to know where to write, avoiding path duplication with the runner.
    stage_output_dir: Optional[Path] = None

    # Source tracking — which stage last produced each artifact
    manifest_producer: str = ""
    video_dir_producer: str = ""

    # Tracking
    completed_steps: List[str] = field(default_factory=list)
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
