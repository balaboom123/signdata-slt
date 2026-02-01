"""How2Sign dataset definition."""

from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("how2sign")
class How2SignDataset(BaseDataset):
    name = "how2sign"

    @classmethod
    def validate_config(cls, config) -> None:
        steps = config.pipeline.steps
        if "download" in steps or "manifest" in steps:
            raise ValueError(
                "how2sign does not support 'download' or 'manifest' steps"
            )
