"""Obfuscate processor: blur or pixelate faces for privacy.

Reads video files from ``context.video_dir``, detects faces using
MediaPipe Face Detection, applies blur or pixelation to face regions,
and writes obfuscated videos to ``{root}/obfuscated/{run_name}/``.

Config via ``stage_config["obfuscate"]``::

    stage_config:
      obfuscate:
        method: blur                    # blur | pixelate
        blur_strength: 51              # Gaussian kernel size (odd int)
        pixelate_size: 10              # block size for pixelation
        min_detection_confidence: 0.5  # MediaPipe face detection threshold
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel, field_validator
from tqdm import tqdm

from .base import BaseProcessor
from ..registry import register_processor
from ..utils.manifest import read_manifest, find_video_file


class ObfuscateConfig(BaseModel):
    """Typed config for the obfuscate stage."""

    method: Literal["blur", "pixelate"] = "blur"
    blur_strength: int = 51
    pixelate_size: int = 10
    min_detection_confidence: float = 0.5

    @field_validator("blur_strength")
    @classmethod
    def _validate_blur_strength(cls, v: int) -> int:
        if v < 1 or v % 2 == 0:
            raise ValueError("blur_strength must be a positive odd integer")
        return v

    @field_validator("pixelate_size")
    @classmethod
    def _validate_pixelate_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("pixelate_size must be a positive integer")
        return v


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _obfuscate_single_video(args) -> Tuple[str, bool, str]:
    """Detect faces and apply obfuscation to a single video.

    Args:
        args: tuple of (video_path, output_path, method,
              blur_strength, pixelate_size, min_detection_confidence)

    Returns:
        (basename, success, message)
    """
    (
        video_path, output_path,
        method, blur_strength, pixelate_size,
        min_detection_confidence, skip_existing,
    ) = args

    name = os.path.basename(output_path)

    try:
        if skip_existing and os.path.exists(output_path):
            return name, True, "skipped (exists)"

        if not os.path.exists(video_path):
            return name, False, f"video not found: {video_path}"

        import cv2
        import mediapipe as mp

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return name, False, "cannot open video"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            cap.release()
            return name, False, "invalid frame dimensions"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return name, False, "cannot open output writer"

        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # full-range model
            min_detection_confidence=min_detection_confidence,
        )

        faces_found = 0
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * width))
                    y = max(0, int(bbox.ymin * height))
                    w = min(int(bbox.width * width), width - x)
                    h = min(int(bbox.height * height), height - y)

                    if w <= 0 or h <= 0:
                        continue

                    faces_found += 1
                    roi = frame[y : y + h, x : x + w]

                    if method == "blur":
                        blurred = cv2.GaussianBlur(
                            roi, (blur_strength, blur_strength), 0,
                        )
                        frame[y : y + h, x : x + w] = blurred
                    else:  # pixelate
                        small = cv2.resize(
                            roi,
                            (max(1, w // pixelate_size), max(1, h // pixelate_size)),
                        )
                        pixelated = cv2.resize(
                            small, (w, h), interpolation=cv2.INTER_NEAREST,
                        )
                        frame[y : y + h, x : x + w] = pixelated

            writer.write(frame)

        face_detection.close()
        cap.release()
        writer.release()

        return name, True, f"{frames_processed} frames, {faces_found} faces"

    except Exception as e:
        return name, False, str(e)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("obfuscate")
class ObfuscateProcessor(BaseProcessor):
    """Apply face obfuscation (blur/pixelate) to video files.

    Reads config from ``stage_config["obfuscate"]``::

        stage_config:
          obfuscate:
            method: blur          # blur | pixelate
            blur_strength: 51     # Gaussian kernel size (odd integer)
            pixelate_size: 10     # block size for pixelation
            min_detection_confidence: 0.5

    Input:  videos from ``context.video_dir``
    Output: obfuscated videos in ``{root}/obfuscated/{run_name}/``
    """

    name = "obfuscate"

    def run(self, context):
        cfg = self.config
        raw_config = cfg.stage_config.get("obfuscate", {})
        stage_cfg = ObfuscateConfig(**raw_config)

        video_dir = str(context.video_dir)

        if context.stage_output_dir:
            output_dir = str(context.stage_output_dir)
        else:
            output_dir = str(
                Path(cfg.paths.root) / "obfuscated" / cfg.run_name
            )

        os.makedirs(output_dir, exist_ok=True)

        # If a prior success marker exists this is a re-run (config change
        # or --force).  Existing outputs may be stale → must rebuild, not skip.
        skip_existing = not os.path.exists(
            os.path.join(output_dir, "_SUCCESS.json")
        )

        data = read_manifest(str(context.manifest_path), normalize_columns=True)

        # Determine file naming based on upstream producer
        clipped = context.video_dir_producer in ("clip_video", "crop_video")
        id_col = "SAMPLE_ID" if clipped else "VIDEO_ID"

        tasks = []
        seen: set = set()
        missing = 0

        for _, row in data.iterrows():
            file_id = str(row[id_col])
            if file_id in seen:
                continue
            seen.add(file_id)

            video_path = str(find_video_file(video_dir, file_id))
            if not os.path.exists(video_path):
                missing += 1
                continue

            out_path = os.path.join(output_dir, f"{file_id}.mp4")
            tasks.append((
                video_path,
                out_path,
                stage_cfg.method,
                stage_cfg.blur_strength,
                stage_cfg.pixelate_size,
                stage_cfg.min_detection_confidence,
                skip_existing,
            ))

        if missing:
            self.logger.warning(
                "%d videos not found in %s", missing, video_dir,
            )

        if not tasks:
            self.logger.info("No obfuscation tasks to process.")
            context.stats["obfuscate"] = {"total": 0}
            return context

        self.logger.info(
            "Obfuscating %d videos → %s", len(tasks), output_dir,
        )

        success = skip = errors = 0
        with ProcessPoolExecutor(
            max_workers=cfg.processing.max_workers,
        ) as executor:
            futures = {
                executor.submit(_obfuscate_single_video, t): t
                for t in tasks
            }
            with tqdm(total=len(tasks), desc="Obfuscating videos") as pbar:
                for future in as_completed(futures):
                    vname, ok, msg = future.result()
                    if ok:
                        if msg == "skipped (exists)":
                            skip += 1
                        else:
                            success += 1
                    else:
                        errors += 1
                        self.logger.error("Failed: %s — %s", vname, msg)
                    pbar.update(1)

        context.stats["obfuscate"] = {
            "total": len(tasks),
            "success": success,
            "skipped": skip,
            "errors": errors,
        }
        return context

    def validate_inputs(self, context) -> None:
        if not context.video_dir:
            raise RuntimeError(
                "Cannot run obfuscate — video directory is not set. "
                "Run upstream video stages first."
            )
        if not context.manifest_path or not context.manifest_path.exists():
            raise RuntimeError(
                "Cannot run obfuscate — manifest not found at "
                f"'{context.manifest_path}'. Run the manifest stage first."
            )
