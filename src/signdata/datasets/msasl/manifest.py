"""MS-ASL manifest building."""

import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .._shared.availability import apply_availability_policy
from .source import MSASLSourceConfig, SPLITS, extract_video_id, load_split_json


def build(config, source: MSASLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from MS-ASL JSON annotation files."""
    manifest_path = config.paths.manifest
    ann_dir = Path(source.annotations_dir)

    if not ann_dir.exists():
        raise FileNotFoundError(f"MS-ASL annotations_dir not found: {ann_dir}")

    selected_splits = SPLITS if source.split == "all" else (source.split,)
    rows: List[Dict] = []

    for split in selected_splits:
        entries = load_split_json(ann_dir, split)
        for entry in entries:
            video_id = extract_video_id(entry["url"])
            start = float(entry["start_time"])
            end = float(entry["end_time"])
            class_id = int(entry["label"])
            gloss = entry.get("text", "")
            signer_id = str(entry.get("signer_id", ""))
            fps = entry.get("fps", 0)
            source_url = entry["url"]

            sample_id = (
                f"{split}-{video_id}"
                f"-{int(start * 1000)}-{int(end * 1000)}"
                f"-{class_id}"
            )

            row: Dict = {
                "SAMPLE_ID": sample_id,
                "VIDEO_ID": video_id,
                "START": start,
                "END": end,
                "CLASS_ID": class_id,
                "TEXT": gloss,
                "SIGNER_ID": signer_id,
                "FPS": fps,
                "SOURCE_URL": source_url,
                "REL_PATH": f"{video_id}.mp4",
                "SPLIT": split,
            }

            box = entry.get("box")
            if isinstance(box, list) and len(box) == 4:
                # box format: [y0, x0, y1, x1]
                row["BBOX_X1"] = float(box[1])
                row["BBOX_Y1"] = float(box[0])
                row["BBOX_X2"] = float(box[3])
                row["BBOX_Y2"] = float(box[2])
                row["PERSON_DETECTED"] = True

            rows.append(row)

    df = pd.DataFrame(rows)

    if source.subset < 1000:
        before = len(df)
        df = df[df["CLASS_ID"] < source.subset].reset_index(drop=True)
        log.info(
            "Subset filter (<%d classes): %d -> %d rows.",
            source.subset, before, len(df),
        )

    video_dir = config.paths.videos
    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    return df
