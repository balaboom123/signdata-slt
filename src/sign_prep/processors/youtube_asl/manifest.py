"""Manifest processor: build segmented CSV from transcripts."""

import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from ..base import BaseProcessor
from ...registry import register_processor
from ...utils.text import normalize_text


@register_processor("manifest")
class ManifestProcessor(BaseProcessor):
    name = "manifest"

    def run(self, context):
        cfg = self.config
        transcript_dir = cfg.paths.transcripts
        manifest_path = cfg.paths.manifest
        max_text_length = cfg.manifest.max_text_length
        min_duration = cfg.manifest.min_duration
        max_duration = cfg.manifest.max_duration

        # Find all transcript JSON files
        json_files = glob(os.path.join(transcript_dir, "*.json"))

        if not json_files:
            self.logger.warning("No transcript files found in %s", transcript_dir)
            context.manifest_path = Path(manifest_path)
            context.stats["manifest"] = {"videos": 0, "segments": 0}
            return context

        self.logger.info(
            "Processing %d transcript files from %s", len(json_files), transcript_dir
        )

        # Remove existing manifest to start fresh
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        processed_count = 0
        total_segments = 0

        for json_file in tqdm(json_files, desc="Building manifest"):
            video_id = os.path.splitext(os.path.basename(json_file))[0]
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)

                if not transcript_data:
                    continue

                segments = self._process_segments(
                    transcript_data, video_id,
                    max_text_length, min_duration, max_duration,
                )

                if segments:
                    self._save_segments(segments, manifest_path)
                    processed_count += 1
                    total_segments += len(segments)

            except Exception as e:
                self.logger.error("Error processing %s: %s", video_id, e)

        context.manifest_path = Path(manifest_path)
        if os.path.exists(manifest_path):
            context.manifest_df = pd.read_csv(
                manifest_path, delimiter="\t", on_bad_lines="skip"
            )

        context.stats["manifest"] = {
            "videos": processed_count,
            "segments": total_segments,
        }
        self.logger.info(
            "Manifest built: %d videos, %d segments -> %s",
            processed_count, total_segments, manifest_path,
        )
        return context

    def _process_segments(
        self,
        transcripts: List[Dict],
        video_id: str,
        max_text_length: int,
        min_duration: float,
        max_duration: float,
    ) -> List[Dict]:
        processed = []
        idx = 0

        valid = [
            t for t in transcripts
            if "text" in t and "start" in t and "duration" in t
        ]

        for entry in valid:
            text = normalize_text(entry["text"])
            dur = entry["duration"]

            if (
                text
                and len(text) <= max_text_length
                and min_duration <= dur <= max_duration
            ):
                processed.append({
                    "VIDEO_NAME": video_id,
                    "SENTENCE_NAME": f"{video_id}-{idx:03d}",
                    "START_REALIGNED": entry["start"],
                    "END_REALIGNED": entry["start"] + dur,
                    "SENTENCE": text,
                })
                idx += 1

        return processed

    def _save_segments(self, segments: List[Dict], csv_path: str) -> None:
        df = pd.DataFrame(segments)
        mode = "a" if os.path.exists(csv_path) else "w"
        header = not os.path.exists(csv_path)
        df.to_csv(
            csv_path,
            sep="\t",
            mode=mode,
            header=header,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
