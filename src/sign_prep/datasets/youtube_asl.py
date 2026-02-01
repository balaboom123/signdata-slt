"""YouTube-ASL dataset definition."""

from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("youtube_asl")
class YouTubeASLDataset(BaseDataset):
    name = "youtube_asl"

    @classmethod
    def validate_config(cls, config) -> None:
        if not config.download.video_ids_file:
            raise ValueError("youtube_asl requires download.video_ids_file")
