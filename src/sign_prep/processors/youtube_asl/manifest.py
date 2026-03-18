"""Manifest processor — backward-compatible wrapper.

Delegates to the dataset adapter's ``build_manifest()`` method.
This wrapper is temporary and will be deleted in Phase 3.
"""

from ..base import BaseProcessor
from ...registry import register_processor


@register_processor("manifest")
class ManifestProcessor(BaseProcessor):
    name = "manifest"

    def run(self, context):
        return context.dataset.build_manifest(self.config, context)
