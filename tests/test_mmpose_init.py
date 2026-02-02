"""Tests for MMPoseExtractor.__init__ parameter handling.

Only tests __init__ parameter handling — no GPU, mmpose, or real frames needed.
"""

from sign_prep.config.schema import ExtractorConfig
from sign_prep.extractors.mmpose import MMPoseExtractor


class TestMMPoseInit:
    def test_default_config(self):
        """Default ExtractorConfig creates extractor with expected attributes."""
        cfg = ExtractorConfig()
        ext = MMPoseExtractor(cfg)
        assert ext.bbox_threshold == 0.5
        assert ext.add_visible is True
        assert ext.det_cat_id == 0

    def test_custom_bbox_threshold(self):
        """Custom bbox_threshold is stored."""
        cfg = ExtractorConfig(bbox_threshold=0.7)
        ext = MMPoseExtractor(cfg)
        assert ext.bbox_threshold == 0.7

    def test_no_reduction_attributes(self):
        """Extractor no longer has reduction-related attributes."""
        cfg = ExtractorConfig()
        ext = MMPoseExtractor(cfg)
        assert not hasattr(ext, "apply_reduction")
        assert not hasattr(ext, "keypoint_indices")
