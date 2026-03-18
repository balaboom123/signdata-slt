"""How2Sign dataset adapter.

How2Sign requires pre-downloaded data and provides official re-aligned
CSV manifests.  The adapter validates that required files exist and
loads the existing manifest.

Source config (Phase 2 bridge — reads from config.paths):
    manifest_csv: str — path to existing re-aligned CSV
    split: str        — which split this CSV represents (train/val/test/all)
"""

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from .base import DatasetAdapter
from ..registry import register_dataset
from ..utils.manifest import read_manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed source config
# ---------------------------------------------------------------------------

class How2SignSourceConfig(BaseModel):
    """Typed config for How2Sign adapter.

    In Phase 2 this is built from existing config fields via
    ``get_source_config()``.  In Phase 3 it will be parsed directly from
    ``config.source``.
    """
    manifest_csv: str = ""
    split: str = "all"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_dataset("how2sign")
class How2SignDataset(DatasetAdapter):
    name = "how2sign"

    @classmethod
    def validate_config(cls, config) -> None:
        steps = config.pipeline.steps
        if "download" in steps:
            raise ValueError(
                "how2sign does not support the 'download' step. "
                "How2Sign requires pre-downloaded data. "
                "See https://how2sign.github.io/"
            )

    def get_source_config(self, config) -> How2SignSourceConfig:
        """Build typed source config from existing config fields (Phase 2 bridge)."""
        return How2SignSourceConfig(
            manifest_csv=config.paths.manifest,
        )

    def acquire(self, config, context):
        """Validate that How2Sign video directory exists.

        How2Sign data must be manually downloaded — this step only checks
        that the expected directories are in place.
        """
        video_dir = config.paths.videos

        if not video_dir:
            raise ValueError(
                "paths.videos is required for How2Sign. "
                "Set it in your config YAML."
            )

        if not Path(video_dir).exists():
            raise FileNotFoundError(
                f"How2Sign video directory not found: {video_dir}\n"
                f"How2Sign requires manual download. "
                f"See https://how2sign.github.io/ for instructions."
            )

        self.logger.info("How2Sign video directory validated: %s", video_dir)
        context.stats["acquire"] = {"validated": True}
        return context

    def build_manifest(self, config, context):
        """Load the existing How2Sign manifest CSV.

        How2Sign provides official re-aligned CSV files; this adapter
        reads one and sets it on the pipeline context.  No column
        renaming is done in Phase 2 (downstream processors still expect
        legacy names like VIDEO_NAME, SENTENCE_NAME, etc.).
        """
        source = self.get_source_config(config)
        manifest_path = source.manifest_csv or config.paths.manifest

        if not manifest_path or not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"How2Sign manifest not found: {manifest_path}\n"
                f"Provide a valid manifest path via paths.manifest in config."
            )

        df = read_manifest(manifest_path, normalize_columns=True)

        context.manifest_path = Path(manifest_path)
        context.manifest_df = df

        context.stats["manifest"] = {
            "videos": df["VIDEO_ID"].nunique() if "VIDEO_ID" in df.columns else 0,
            "segments": len(df),
        }
        self.logger.info(
            "How2Sign manifest loaded: %d segments from %s",
            len(df), manifest_path,
        )
        return context
