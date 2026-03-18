"""Crop video processor: crop clipped videos to the detected person bbox.

Reads BBOX_* and PERSON_DETECTED columns from the manifest (written by
detect_person), applies padding, clamps to frame boundaries, and
re-encodes using ffmpeg.

If PERSON_DETECTED is False, the clip is copied as-is (no crop).
Output goes to paths.cropped_clips, leaving paths.clips untouched.
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

from ..base import BaseProcessor
from ...registry import register_processor
from ...utils.manifest import read_manifest


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _crop_single_video(args) -> Tuple[str, bool, str]:
    """Crop (or copy) a single video clip.

    Args:
        args: tuple of
            (clip_path, output_path, x1, y1, x2, y2,
             person_detected, padding, codec)

    Returns:
        (name, success, message)
    """
    (
        clip_path, output_path,
        x1, y1, x2, y2,
        person_detected,
        padding, codec,
    ) = args

    name = os.path.basename(output_path)

    try:
        if os.path.exists(output_path):
            return name, True, "skipped (exists)"

        if not os.path.exists(clip_path):
            return name, False, f"clip not found: {clip_path}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Resolve ffmpeg to its full path so worker processes on Windows
        # can find it even when PATH is not fully inherited.
        import shutil as _shutil
        ffmpeg_bin = _shutil.which("ffmpeg") or "ffmpeg"

        # If no person detected, stream-copy without cropping
        if not person_detected:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", clip_path,
                "-c", "copy",
                "-an", output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return name, False, result.stderr[-300:]
            return name, True, "no-person copy"

        # Read frame size from the clip itself
        video_capture = cv2.VideoCapture(clip_path)
        if not video_capture.isOpened():
            return name, False, "cannot open clip to read dimensions"
        frame_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()

        if frame_w == 0 or frame_h == 0:
            return name, False, "invalid frame dimensions"

        # Apply padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = bbox_w * padding
        pad_y = bbox_h * padding

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w, x2 + pad_x)
        cy2 = min(frame_h, y2 + pad_y)

        # Convert to integers; ffmpeg crop requires even dimensions for
        # most codecs, so round down to nearest even number.
        cx1 = int(cx1)
        cy1 = int(cy1)
        crop_w = int(cx2 - cx1)
        crop_h = int(cy2 - cy1)

        # Ensure even dimensions (required by libx264 and most H.264 encoders)
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)

        if crop_w <= 0 or crop_h <= 0:
            return name, False, f"degenerate crop region: w={crop_w} h={crop_h}"

        vf = f"crop={crop_w}:{crop_h}:{cx1}:{cy1}"

        cmd = [
            ffmpeg_bin, "-y",
            "-i", clip_path,
            "-vf", vf,
            "-c:v", codec,
            "-an",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return name, False, result.stderr[-300:]

        return name, True, ""

    except subprocess.TimeoutExpired:
        return name, False, "ffmpeg timeout"
    except Exception as e:
        return name, False, str(e)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("crop_video")
class CropVideoProcessor(BaseProcessor):
    """Re-encode clipped videos cropped to the detected person bbox.

    Reads the manifest for BBOX_* columns written by DetectPersonProcessor.
    Input:  paths.clips      (produced by clip_video)
    Output: paths.cropped_clips
    """

    name = "crop_video"

    def run(self, context):
        cfg = self.config
        crop_cfg = cfg.crop_video
        manifest_path = cfg.paths.manifest
        clips_dir = cfg.paths.clips
        cropped_dir = cfg.paths.cropped_clips

        if not cropped_dir:
            raise ValueError(
                "paths.cropped_clips is not set in config. "
                "Please add it to your yaml (e.g. dataset/youtube_asl/cropped_clips)."
            )

        os.makedirs(cropped_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # Load manifest — must have bbox columns from detect_person
        # ----------------------------------------------------------------
        data = read_manifest(manifest_path, normalize_columns=True)

        required_cols = {"BBOX_X1", "BBOX_Y1", "BBOX_X2", "BBOX_Y2", "PERSON_DETECTED"}
        missing = required_cols - set(data.columns)
        if missing:
            raise RuntimeError(
                f"Manifest is missing columns: {missing}. "
                "Run the 'detect_person' step first."
            )

        data = data[
            ["SAMPLE_ID", "BBOX_X1", "BBOX_Y1", "BBOX_X2", "BBOX_Y2", "PERSON_DETECTED"]
        ].dropna(subset=["SAMPLE_ID"])

        # ----------------------------------------------------------------
        # Build task list
        # ----------------------------------------------------------------
        tasks = []
        missing_clips = 0

        for _, row in data.iterrows():
            clip_path = os.path.join(clips_dir, f"{row.SAMPLE_ID}.mp4")
            out_path = os.path.join(cropped_dir, f"{row.SAMPLE_ID}.mp4")

            if not os.path.exists(clip_path):
                missing_clips += 1
                continue

            # PERSON_DETECTED may have been stored as string "True"/"False"
            person_detected = _parse_bool(row["PERSON_DETECTED"])

            tasks.append((
                clip_path, out_path,
                float(row["BBOX_X1"]), float(row["BBOX_Y1"]),
                float(row["BBOX_X2"]), float(row["BBOX_Y2"]),
                person_detected,
                crop_cfg.padding,
                crop_cfg.codec,
            ))

        if missing_clips:
            self.logger.warning(
                "%d clips not found in %s — run clip_video first.",
                missing_clips, clips_dir,
            )

        if not tasks:
            self.logger.info("No crop tasks to process.")
            context.stats["crop_video"] = {"total": 0}
            return context

        self.logger.info("Cropping %d clips → %s", len(tasks), cropped_dir)

        # ----------------------------------------------------------------
        # Parallel execution
        # ----------------------------------------------------------------
        success = skip = no_person_copy = errors = 0

        with ProcessPoolExecutor(max_workers=cfg.processing.max_workers) as executor:
            futures = {executor.submit(_crop_single_video, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Cropping videos") as pbar:
                for future in as_completed(futures):
                    name, ok, msg = future.result()
                    if ok:
                        if msg == "skipped (exists)":
                            skip += 1
                        elif msg == "no-person copy":
                            no_person_copy += 1
                        else:
                            success += 1
                    else:
                        errors += 1
                        self.logger.error("Failed: %s — %s", name, msg)
                    pbar.update(1)

        context.stats["crop_video"] = {
            "total": len(tasks),
            "cropped": success,
            "copied_no_person": no_person_copy,
            "skipped": skip,
            "errors": errors,
        }
        return context


def _parse_bool(val) -> bool:
    """Parse a boolean that may be stored as bool, str, int, or float."""
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        return bool(val)  # handles NaN → False gracefully
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)