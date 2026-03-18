"""Download processor — backward-compatible wrapper.

Delegates to the dataset adapter's ``acquire()`` method.
This wrapper is temporary and will be deleted in Phase 3.
"""

from ..base import BaseProcessor
from ...registry import register_processor


@register_processor("download")
class DownloadProcessor(BaseProcessor):
    name = "download"

    def run(self, context):
        return context.dataset.acquire(self.config, context)
