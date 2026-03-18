"""WebDataset processor: package outputs into tar shards."""

import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BaseProcessor
from ...registry import register_processor
from ...utils.manifest import read_manifest, get_timing_columns


class _ShardWriter:
    """Minimal shard writer using Python's tarfile module.

    Replaces webdataset.ShardWriter to avoid the gopen() Windows path issue
    where drive letters (e.g. D:/) are misinterpreted as URL schemes.

    Produces tar files that are fully compatible with webdataset readers.
    """

    def __init__(
        self,
        output_dir: str,
        max_count: int = 10_000,
        max_size: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.max_count = max_count
        self.max_size = max_size  # bytes; None = no limit

        self._shard_idx = 0
        self._count = 0
        self._size = 0
        self._tar: Optional[tarfile.TarFile] = None
        self._current_path: Optional[Path] = None
        self._open_shard()

    def _shard_path(self) -> Path:
        return self.output_dir / f"shard-{self._shard_idx:06d}.tar"

    def _open_shard(self):
        if self._tar is not None:
            self._tar.close()
        self._current_path = self._shard_path()
        self._tar = tarfile.open(str(self._current_path), "w")
        self._count = 0
        self._size = 0

    def _next_shard(self):
        self._tar.close()
        self._shard_idx += 1
        self._open_shard()

    def _add_bytes(self, name: str, data: bytes):
        """Add a single file entry to the current tar."""
        buf = io.BytesIO(data)
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        self._tar.addfile(info, buf)
        self._size += len(data)

    def write(self, sample: dict):
        """Write one webdataset sample.

        sample must contain "__key__" and any number of extension→bytes pairs.
        String values are UTF-8 encoded automatically.
        """
        key = sample["__key__"]

        # Roll over shard if needed (check before writing).
        # Use elif to prevent double _next_shard() when both limits are hit at once,
        # which would create an empty intermediate shard file.
        if self._count >= self.max_count:
            self._next_shard()
        elif self.max_size and self._size >= self.max_size:
            self._next_shard()

        for ext, value in sample.items():
            if ext == "__key__":
                continue
            if isinstance(value, str):
                value = value.encode("utf-8")
            self._add_bytes(f"{key}.{ext}", value)

        self._count += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    @property
    def shard_count(self) -> int:
        return self._shard_idx + 1


@register_processor("webdataset")
class WebDatasetProcessor(BaseProcessor):
    name = "webdataset"

    def run(self, context):
        cfg = self.config
        mode = cfg.pipeline.mode
        output_dir = cfg.paths.webdataset
        os.makedirs(output_dir, exist_ok=True)

        manifest_path = cfg.paths.manifest
        data = read_manifest(manifest_path, normalize_columns=True)
        start_col, end_col = get_timing_columns(data)

        # Text column (canonical name after normalization)
        sentence_col = "TEXT" if "TEXT" in data.columns else None

        max_count = cfg.webdataset.max_shard_count
        max_size = cfg.webdataset.max_shard_size or None

        # For video mode: prefer cropped_clips if available, fall back to clips.
        if mode == "video":
            cropped_dir = cfg.paths.cropped_clips
            clips_dir = cfg.paths.clips
            if cropped_dir and os.path.isdir(cropped_dir) and any(
                f.endswith(".mp4") for f in os.listdir(cropped_dir)
            ):
                video_source_dir = cropped_dir
                self.logger.info("video mode: using cropped_clips → %s", cropped_dir)
            else:
                video_source_dir = clips_dir
                self.logger.info("video mode: using clips → %s", clips_dir)

        written = 0
        skipped = 0

        with _ShardWriter(output_dir, max_count=max_count, max_size=max_size) as sink:
            for _, row in data.iterrows():
                sample_id = row.SAMPLE_ID
                video_id = row.VIDEO_ID

                caption = ""
                if sentence_col and pd.notna(row.get(sentence_col)):
                    caption = str(row[sentence_col])

                meta = {
                    "video_id": str(video_id),
                    "sample_id": str(sample_id),
                    "start": float(row[start_col]),
                    "end": float(row[end_col]),
                    "extractor": cfg.extractor.name,
                    "mode": mode,
                }

                sample = {"__key__": sample_id}

                if mode == "pose":
                    npy_path = os.path.join(
                        cfg.paths.normalized, f"{sample_id}.npy"
                    )
                    if not os.path.exists(npy_path):
                        skipped += 1
                        continue
                    arr = np.load(npy_path)
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    sample["npy"] = buf.getvalue()

                elif mode == "video":
                    clip_path = os.path.join(
                        video_source_dir, f"{sample_id}.mp4"
                    )
                    if not os.path.exists(clip_path):
                        skipped += 1
                        continue
                    with open(clip_path, "rb") as f:
                        sample["mp4"] = f.read()

                sample["txt"] = caption
                sample["json"] = json.dumps(meta).encode("utf-8")

                sink.write(sample)
                written += 1

        context.stats["webdataset"] = {
            "written": written,
            "skipped": skipped,
            "shards": sink.shard_count,
            "output_dir": output_dir,
        }
        self.logger.info(
            "WebDataset: wrote %d samples in %d shard(s), skipped %d → %s",
            written, sink.shard_count, skipped, output_dir,
        )
        return context