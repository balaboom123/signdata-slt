"""Extraction processor: extract pose landmarks from video segments."""

import gc
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil

from ..base import BaseProcessor
from ...registry import register_processor
from ...utils.video import FPSSampler, validate_video_file, get_video_fps
from ...utils.files import get_video_filenames
from ...utils.manifest import read_manifest, get_timing_columns


def _build_processing_tasks(
    timestamp_data: pd.DataFrame,
    video_dir: str,
    output_dir: str,
    start_col: str,
    end_col: str,
    existing_files: Optional[List[str]] = None,
    min_duration: float = 0.2,
    max_duration: float = 60.0,
    fps_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[str, float, float, str]], Dict[str, int]]:
    """Build task list from manifest with validation and filtering."""
    import logging
    logger = logging.getLogger("sign_prep.extract")

    existing_set = set(existing_files or [])
    video_validation_cache: Dict[str, bool] = {}
    video_fps_cache: Dict[str, float] = {}
    invalid_videos: Set[str] = set()

    stats = {
        "existing_files": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_video": 0,
        "fps_out_of_range": 0,
    }

    tasks = []

    for _, row in timestamp_data.iterrows():
        video_name = row.VIDEO_ID
        sentence_name = row.SAMPLE_ID
        start, end = row[start_col], row[end_col]

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        output_path = os.path.join(output_dir, f"{sentence_name}.npy")

        if sentence_name in existing_set:
            stats["existing_files"] += 1
            continue

        seg_dur = float(end - start)
        if seg_dur < min_duration:
            stats["too_short"] += 1
            continue
        if seg_dur > max_duration:
            stats["too_long"] += 1
            continue

        if video_path not in video_validation_cache:
            video_validation_cache[video_path] = validate_video_file(video_path)
            if not video_validation_cache[video_path]:
                invalid_videos.add(video_name)

        if not video_validation_cache[video_path]:
            stats["invalid_video"] += 1
            continue

        if fps_range is not None:
            if video_path not in video_fps_cache:
                video_fps_cache[video_path] = get_video_fps(video_path)
            video_fps = video_fps_cache[video_path]
            min_fps, max_fps = fps_range
            if video_fps <= 0.0 or video_fps < float(min_fps) or video_fps > float(max_fps):
                stats["fps_out_of_range"] += 1
                continue

        tasks.append((video_path, start, end, output_path))

    logger.info("Task summary:")
    logger.info("  Tasks to process: %d", len(tasks))
    for key, val in stats.items():
        logger.info("  Skipped (%s): %d", key, val)

    return tasks, stats


# --- Worker globals for MMPose ---
_detector = None
_pose_estimator = None
_worker_config_dict = None


def _iter_batches_with_sampling(
    video_capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    sampler: FPSSampler,
    batch_size: int,
) -> Generator[List[np.ndarray], None, None]:
    """Yield batches of sampled frames from video segment.

    Memory-efficient: only holds batch_size frames at a time.
    """
    batch = []
    current = start_frame
    while current <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        if sampler.take():
            batch.append(frame)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        current += 1
    if batch:  # Yield remaining frames
        yield batch


def _init_mmpose_worker(config_dict: dict):
    """Initialize MMPose models once per worker process."""
    global _detector, _pose_estimator, _worker_config_dict
    _worker_config_dict = config_dict

    from mmpose.apis import init_model
    from mmpose.utils import adapt_mmdet_pipeline
    from mmdet.apis import init_detector

    device = config_dict["device"]
    _detector = init_detector(
        config_dict["det_model_config"],
        config_dict["det_model_checkpoint"],
        device=device,
    )
    _detector.cfg = adapt_mmdet_pipeline(_detector.cfg)

    _pose_estimator = init_model(
        config_dict["pose_model_config"],
        config_dict["pose_model_checkpoint"],
        device=device,
    )
    _pose_estimator.cfg.model.test_cfg.mode = "3d"


def _process_segment_mediapipe(args):
    """Process a single video segment with MediaPipe using batch processing."""
    video_path, start_time, end_time, output_file, config_dict = args
    import logging
    logger = logging.getLogger("sign_prep.extract")

    from ...config.schema import ExtractorConfig
    from ...extractors.mediapipe import MediaPipeExtractor

    video_capture = None
    extractor = None
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return

        fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
        target_fps = config_dict.get("target_fps")
        frame_skip = config_dict.get("frame_skip", 2)
        sampler = FPSSampler(src_fps=fps, reduce_to=target_fps, frame_skip_by=frame_skip)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ext_config = ExtractorConfig(**config_dict["extractor"])
        extractor = MediaPipeExtractor(ext_config)
        batch_size = ext_config.batch_size

        num_landmarks = extractor.num_landmarks

        # Process frames in rolling batches (bounded memory)
        sequences = []
        for batch_frames in _iter_batches_with_sampling(
            video_capture, start_frame, end_frame, sampler, batch_size
        ):
            batch_results = extractor.process_batch(
                batch_frames, fallback_on_error=True
            )
            for landmarks in batch_results:
                if landmarks is None:
                    landmarks = np.zeros((num_landmarks, 4), dtype=np.float32)
                sequences.append(landmarks)

        if sequences:
            arr = np.array(sequences)
            if arr.size > 0 and np.any(arr):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, arr)

    except Exception as e:
        logger.error("Error processing %s: %s", video_path, e)
    finally:
        if video_capture is not None:
            video_capture.release()
        if extractor is not None:
            extractor.close()
        gc.collect()


def _process_segment_mmpose(args):
    """Process a single video segment with MMPose using batch processing."""
    video_path, start_time, end_time, output_file, config_dict = args
    import logging
    logger = logging.getLogger("sign_prep.extract")

    global _detector, _pose_estimator

    from ...config.schema import ExtractorConfig
    from ...extractors.mmpose import MMPoseExtractor

    video_capture = None
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return

        fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
        target_fps = config_dict.get("target_fps")
        frame_skip = config_dict.get("frame_skip", 2)
        sampler = FPSSampler(src_fps=fps, reduce_to=target_fps, frame_skip_by=frame_skip)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ext_config = ExtractorConfig(**config_dict["extractor"])
        extractor = MMPoseExtractor(
            config=ext_config,
            detector=_detector,
            pose_estimator=_pose_estimator,
        )
        batch_size = ext_config.batch_size

        num_landmarks = extractor.num_landmarks

        # Process frames in rolling batches (bounded memory)
        # Multi-person frames return None and are handled gracefully
        sequences = []

        for batch_frames in _iter_batches_with_sampling(
            video_capture, start_frame, end_frame, sampler, batch_size
        ):
            batch_results = extractor.process_batch(
                batch_frames, fallback_on_error=True
            )

            for landmarks in batch_results:
                if landmarks is None:
                    # Could be no detection or multi-person
                    # We continue processing; use zeros for those frames
                    landmarks = np.zeros((num_landmarks, 4), dtype=np.float32)
                sequences.append(landmarks)

        # Only save if we have valid sequences
        # Note: We no longer abort on multi-person; we use zeros for those frames
        if sequences:
            arr = np.array(sequences)
            if arr.size > 0 and np.any(arr):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, arr)

    except Exception as e:
        logger.error("Error processing %s: %s", video_path, e)
    finally:
        if video_capture is not None:
            video_capture.release()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


@register_processor("extract")
class ExtractProcessor(BaseProcessor):
    name = "extract"

    def run(self, context):
        cfg = self.config
        manifest_path = cfg.paths.manifest

        # Load manifest
        if context.manifest_df is not None:
            timestamp_data_full = context.manifest_df
        else:
            timestamp_data_full = read_manifest(manifest_path, normalize_columns=True)

        start_col, end_col = get_timing_columns(timestamp_data_full)
        timestamp_data = timestamp_data_full[
            ["VIDEO_ID", "SAMPLE_ID", start_col, end_col]
        ].dropna()

        landmarks_dir = cfg.paths.landmarks
        os.makedirs(landmarks_dir, exist_ok=True)

        existing = get_video_filenames(landmarks_dir, pattern="*.npy")

        fps_range = None
        if cfg.processing.accept_fps_range:
            fps_range = tuple(cfg.processing.accept_fps_range)

        tasks, stats = _build_processing_tasks(
            timestamp_data=timestamp_data,
            video_dir=cfg.paths.videos,
            output_dir=landmarks_dir,
            start_col=start_col,
            end_col=end_col,
            existing_files=existing,
            min_duration=cfg.processing.min_duration,
            max_duration=cfg.processing.max_duration,
            fps_range=fps_range,
        )

        if not tasks:
            self.logger.info("No extraction tasks to process.")
            context.stats["extract"] = stats
            return context

        # Build serializable config dict for workers
        config_dict = {
            "target_fps": cfg.processing.target_fps,
            "frame_skip": cfg.processing.frame_skip,
            "device": cfg.extractor.device,
            "extractor": cfg.extractor.model_dump(),
        }

        extractor_name = cfg.extractor.name
        max_workers = min(
            cfg.extractor.max_workers,
            max(multiprocessing.cpu_count() - 1, 1),
        )

        self.logger.info(
            "Extracting with %s, %d workers, %d tasks",
            extractor_name, max_workers, len(tasks),
        )

        if extractor_name == "mediapipe":
            self._run_mediapipe(tasks, config_dict, max_workers)
        elif extractor_name == "mmpose":
            self._run_mmpose(tasks, config_dict, max_workers)
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")

        context.stats["extract"] = {**stats, "tasks_submitted": len(tasks)}
        return context

    def _run_mediapipe(self, tasks, config_dict, max_workers):
        BATCH_SIZE = 100
        args_list = [
            (vp, s, e, op, config_dict) for vp, s, e, op in tasks
        ]

        for i in range(0, len(args_list), BATCH_SIZE):
            batch = args_list[i : i + BATCH_SIZE]
            self.logger.info(
                "Batch %d: tasks %d-%d",
                i // BATCH_SIZE + 1, i + 1, min(i + BATCH_SIZE, len(args_list)),
            )

            per_worker = len(batch) // max_workers + 1
            worker_batches = [
                batch[j : j + per_worker]
                for j in range(0, len(batch), per_worker)
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for wb in worker_batches:
                    for item in wb:
                        futures.append(
                            executor.submit(_process_segment_mediapipe, item)
                        )
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:
                        self.logger.error("Worker error: %s", e)

            time.sleep(0.5)
            gc.collect()

    def _run_mmpose(self, tasks, config_dict, max_workers):
        BATCH_SIZE = 128
        args_list = [
            (vp, s, e, op, config_dict) for vp, s, e, op in tasks
        ]

        # MMPose config for worker init
        mmpose_init_dict = {
            "det_model_config": self.config.extractor.det_model_config,
            "det_model_checkpoint": self.config.extractor.det_model_checkpoint,
            "pose_model_config": self.config.extractor.pose_model_config,
            "pose_model_checkpoint": self.config.extractor.pose_model_checkpoint,
            "device": self.config.extractor.device,
        }

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_mmpose_worker,
            initargs=(mmpose_init_dict,),
        ) as executor:
            for i in range(0, len(args_list), BATCH_SIZE):
                batch = args_list[i : i + BATCH_SIZE]
                self.logger.info(
                    "Batch %d: tasks %d-%d",
                    i // BATCH_SIZE + 1,
                    i + 1,
                    min(i + BATCH_SIZE, len(args_list)),
                )

                futures = [
                    executor.submit(_process_segment_mmpose, item)
                    for item in batch
                ]
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:
                        self.logger.error("Worker error: %s", e)

                time.sleep(0.5)
                gc.collect()
