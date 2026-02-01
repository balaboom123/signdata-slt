"""WebDataset processor: package outputs into tar shards."""

import io
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import webdataset as wds

from ..base import BaseProcessor
from ...registry import register_processor


def _read_manifest_csv(csv_file: str) -> Tuple[pd.DataFrame, str, str]:
    """Read manifest CSV and detect timestamp column format."""
    data = pd.read_csv(csv_file, delimiter="\t", on_bad_lines="skip")
    columns = data.columns.tolist()
    if "START" in columns and "END" in columns:
        return data, "START", "END"
    elif "START_REALIGNED" in columns and "END_REALIGNED" in columns:
        return data, "START_REALIGNED", "END_REALIGNED"
    raise ValueError("No recognized timestamp columns found")


@register_processor("webdataset")
class WebDatasetProcessor(BaseProcessor):
    name = "webdataset"

    def run(self, context):
        cfg = self.config
        mode = cfg.pipeline.mode
        output_dir = cfg.paths.webdataset
        os.makedirs(output_dir, exist_ok=True)

        manifest_path = cfg.paths.manifest
        data, start_col, end_col = _read_manifest_csv(manifest_path)

        # Build sentence lookup
        sentence_col = None
        for col in ["SENTENCE", "TEXT", "CAPTION"]:
            if col in data.columns:
                sentence_col = col
                break

        shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
        max_count = cfg.webdataset.max_shard_count
        max_size = cfg.webdataset.max_shard_size

        writer_kwargs = {"pattern": shard_pattern, "maxcount": max_count}
        if max_size:
            writer_kwargs["maxsize"] = max_size

        written = 0
        skipped = 0

        with wds.ShardWriter(**writer_kwargs) as sink:
            for _, row in data.iterrows():
                sentence_name = row.SENTENCE_NAME
                video_name = row.VIDEO_NAME

                caption = ""
                if sentence_col and pd.notna(row.get(sentence_col)):
                    caption = str(row[sentence_col])

                meta = {
                    "video_id": str(video_name),
                    "sentence_name": str(sentence_name),
                    "start": float(row[start_col]),
                    "end": float(row[end_col]),
                    "extractor": cfg.extractor.name,
                    "mode": mode,
                }

                sample = {"__key__": sentence_name}

                if mode == "pose":
                    npy_path = os.path.join(
                        cfg.paths.normalized, f"{sentence_name}.npy"
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
                        cfg.paths.clips, f"{sentence_name}.mp4"
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
            "output_dir": output_dir,
        }
        self.logger.info(
            "WebDataset: wrote %d samples, skipped %d -> %s",
            written, skipped, output_dir,
        )
        return context
