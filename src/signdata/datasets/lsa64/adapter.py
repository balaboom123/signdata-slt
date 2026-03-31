"""LSA64 dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("lsa64")
class LSA64Dataset(DatasetAdapter):
    """LSA64 Argentinian Sign Language dataset adapter."""

    name = "lsa64"

    @classmethod
    def validate_config(cls, config) -> None:
        pass

    def download(self, config, context):
        source = _source.get_source_config(config)
        video_dir = _source.resolve_video_dir(config, source)
        stats = _source.validate_release(source, video_dir, self.logger)
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
            "classes": int(df["CLASS_ID"].nunique()),
            "signers": int(df["SIGNER_ID"].nunique()),
        }
        self.logger.info(
            "LSA64 manifest built: %d segments, %d classes, %d signers -> %s",
            len(df), df["CLASS_ID"].nunique(), df["SIGNER_ID"].nunique(),
            config.paths.manifest,
        )
        return context
