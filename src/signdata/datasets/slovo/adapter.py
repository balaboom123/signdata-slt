"""SLoVo dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("slovo")
class SlovoDataset(DatasetAdapter):
    """SLoVo Russian Sign Language dataset adapter."""

    name = "slovo"

    @classmethod
    def validate_config(cls, config) -> None:
        pass

    def download(self, config, context):
        source = _source.get_source_config(config)
        stats = _source.validate(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = _source.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "split": source.split,
            "class_map_mode": source.class_map_mode,
        }
        self.logger.info(
            "SLoVo manifest built: %d segments, %d unique signers -> %s",
            len(df), df["SIGNER_ID"].nunique(), config.paths.manifest,
        )
        return context
