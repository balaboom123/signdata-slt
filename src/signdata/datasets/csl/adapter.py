"""CSL dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("csl")
class CSLDataset(DatasetAdapter):
    """CSL Chinese Sign Language dataset adapter."""

    name = "csl"

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
            "signers": int(df["SIGNER_ID"].nunique()) if "SIGNER_ID" in df.columns else 0,
            "variant": source.variant,
            "protocol": source.protocol,
        }
        self.logger.info(
            "CSL manifest built: %d segments, %d signers -> %s",
            len(df), df["SIGNER_ID"].nunique(), config.paths.manifest,
        )
        return context
