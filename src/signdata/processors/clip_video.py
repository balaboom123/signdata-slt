"""Video clipping processor: segment videos into clips using ffmpeg."""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from .base import BaseProcessor
from ..registry import register_processor
from ..utils.manifest import read_manifest, get_timing_columns, find_video_file


def _clip_single_video(args) -> Tuple[str, bool, str]:
    """Clip a single video segment using ffmpeg."""
    video_path, start, end, output_path, codec, resize = args
    name = os.path.basename(output_path)

    try:
        if os.path.exists(output_path):
            return name, True, "skipped (exists)"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        duration = end - start

        import shutil
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
        ]

        if codec == "copy":
            cmd.extend(["-c", "copy"])
        else:
            cmd.extend(["-c:v", codec])
            if resize:
                cmd.extend(["-vf", f"scale={resize[0]}:{resize[1]}"])

        cmd.extend(["-an", output_path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            return name, False, result.stderr[-200:]

        return name, True, ""
    except subprocess.TimeoutExpired:
        return name, False, "ffmpeg timeout"
    except Exception as e:
        return name, False, str(e)


@register_processor("clip_video")
class ClipVideoProcessor(BaseProcessor):
    name = "clip_video"

    def run(self, context):
        cfg = self.config
        manifest_path = str(context.manifest_path)
        video_dir = str(context.video_dir)
        clips_dir = cfg.paths.clips

        os.makedirs(clips_dir, exist_ok=True)

        data = read_manifest(manifest_path, normalize_columns=True)
        start_col, end_col = get_timing_columns(data)
        data = data[["VIDEO_ID", "SAMPLE_ID", start_col, end_col]].dropna()

        codec = cfg.clip_video.codec
        resize = cfg.clip_video.resize

        tasks = []
        for _, row in data.iterrows():
            vpath = str(find_video_file(video_dir, row.VIDEO_ID))
            opath = os.path.join(clips_dir, f"{row.SAMPLE_ID}.mp4")
            if not os.path.exists(vpath):
                continue
            tasks.append((
                vpath, float(row[start_col]), float(row[end_col]),
                opath, codec, resize,
            ))

        if not tasks:
            self.logger.info("No clip tasks to process.")
            context.stats["clip_video"] = {"total": 0}
            return context

        self.logger.info("Clipping %d segments", len(tasks))

        success = skip = errors = 0
        with ProcessPoolExecutor(max_workers=cfg.processing.max_workers) as executor:
            futures = {executor.submit(_clip_single_video, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Clipping videos") as pbar:
                for future in as_completed(futures):
                    name, ok, msg = future.result()
                    if ok:
                        if msg == "skipped (exists)":
                            skip += 1
                        else:
                            success += 1
                    else:
                        errors += 1
                        self.logger.error("Failed: %s - %s", name, msg)
                    pbar.update(1)

        context.stats["clip_video"] = {
            "total": len(tasks),
            "success": success,
            "skipped": skip,
            "errors": errors,
        }
        return context
