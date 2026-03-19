"""Smoke test for detect_person and crop_video processors.

Usage:
    python scripts/smoke_test_localize.py --video path/to/any_video.mp4

This script:
  1. Creates a temporary workspace with a minimal manifest
  2. Runs DetectPersonProcessor  → writes BBOX_* columns to manifest
  3. Runs ClipVideoProcessor       → clips the segment
  4. Runs CropVideoProcessor       → crops the clip to the detected person
  5. Prints a summary and shows where to find the output files

No dataset download required — just supply any .mp4 file.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import cv2
import pandas as pd

# Make sure src/ is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_video_duration(video_path: str) -> float:
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_capture.release()
    if fps > 0:
        return frame_count / fps
    return 0.0


def build_manifest(video_path: str, workspace: Path, num_segments: int, segment_duration: float) -> Path:
    """Create a manifest by evenly spacing num_segments clips across the video."""
    duration = get_video_duration(video_path)
    if duration < segment_duration:
        raise ValueError(f"Video too short ({duration:.1f}s) for segment_duration={segment_duration}s.")

    video_name = Path(video_path).stem

    # Space segment start points evenly across the video
    usable = duration - segment_duration
    starts = [i * usable / max(num_segments - 1, 1) for i in range(num_segments)]

    rows = {
        "VIDEO_NAME":      [],
        "SENTENCE_NAME":   [],
        "START_REALIGNED": [],
        "END_REALIGNED":   [],
        "SENTENCE":        [],
    }
    for i, start in enumerate(starts):
        end = start + segment_duration
        rows["VIDEO_NAME"].append(video_name)
        rows["SENTENCE_NAME"].append(f"{video_name}-{i}")
        rows["START_REALIGNED"].append(round(start, 3))
        rows["END_REALIGNED"].append(round(end, 3))
        rows["SENTENCE"].append(f"segment {i}")
        print(f"  Segment {i}: {start:.1f}s → {end:.1f}s")

    manifest_path = workspace / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    print(f"  Created manifest: {manifest_path}")
    return manifest_path


def run_smoke_test(video_path: str, device: str, padding: float, max_frames: int,
                   output_dir: str = None, num_segments: int = 2, segment_duration: float = 10.0):
    import shutil

    # Pre-flight: verify ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("[ERROR] ffmpeg not found in PATH.")
        print("  Install ffmpeg and make sure it is accessible:")
        print("  Windows: https://www.gyan.dev/ffmpeg/builds/")
        print("  Or via conda: conda install -c conda-forge ffmpeg")
        sys.exit(1)
    print(f"  ffmpeg : {ffmpeg_path}")

    from sign_prep.config.schema import Config
    from sign_prep.pipeline.context import PipelineContext
    from sign_prep.datasets.youtube_asl import YouTubeASLDataset
    from sign_prep.processors.detect_person import DetectPersonProcessor
    from sign_prep.processors.clip_video import ClipVideoProcessor
    from sign_prep.processors.crop_video import CropVideoProcessor

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Smoke test: detect_person + clip_video + crop_video")
    print(f"  Video : {video_path}")
    print(f"  Device: {device}  |  Padding: {padding}  |  Max frames: {max_frames}  |  Segments: {num_segments} × {segment_duration}s")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # Setup workspace — mirrors dataset/how2sign layout
    # ----------------------------------------------------------------
    if output_dir:
        workspace = Path(output_dir)
    else:
        workspace = PROJECT_ROOT / "dataset" / "smoke_test"

    videos_dir  = workspace / "videos"
    clips_dir   = workspace / "clips"
    cropped_dir = workspace / "cropped_clips"
    videos_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    # Symlink (or copy) the video into the videos dir so the processor finds it.
    # Skip if the video is already inside videos_dir (e.g. re-running with the same path).
    video_name = Path(video_path).stem
    dest_video = videos_dir / f"{video_name}.mp4"
    if Path(video_path).resolve() != dest_video.resolve():
        try:
            os.symlink(video_path, dest_video)
        except (OSError, NotImplementedError):
            # Windows may not support symlinks without admin rights → copy instead
            shutil.copy2(video_path, dest_video)

    print(f"[1/4] Building manifest ({num_segments} segments × {segment_duration}s)...")
    manifest_path = build_manifest(str(dest_video), workspace, num_segments, segment_duration)

    # ----------------------------------------------------------------
    # Build config
    # ----------------------------------------------------------------
    cfg = Config(
        dataset="youtube_asl",
        recipe="video",
        source={"video_ids_file": "/dev/null"},
        paths={
            "root":          str(workspace),
            "videos":        str(videos_dir),
            "manifest":      str(manifest_path),
            "clips":         str(clips_dir),
            "cropped_clips": str(cropped_dir),
        },
        detect_person={
            "enabled":              True,
            "model":                "yolov8n.pt",
            "confidence_threshold": 0.5,
            "max_frames":           max_frames,
            "uniform_frames":       max_frames,
            "device":               device,
            "min_bbox_area":        0.02,   # relaxed for smoke test
        },
        crop_video={
            "enabled": True,
            "padding": padding,
            "codec":   "libx264",
        },
        clip_video={
            "codec": "libx264",   # re-encode so crop filter can be applied after
        },
        processing={"max_workers": 2},
    )

    ctx = PipelineContext(
        config=cfg,
        dataset=YouTubeASLDataset(),
        project_root=workspace,
        manifest_path=manifest_path,
        video_dir=videos_dir,
    )

    # ----------------------------------------------------------------
    # Step 1: detect_person
    # ----------------------------------------------------------------
    print("\n[2/4] Running detect_person...")
    ctx.stage_output_dir = workspace / "detect_person" / cfg.run_name
    processor = DetectPersonProcessor(cfg)
    ctx = processor.run(ctx)

    # Manual context routing (normally the runner does this)
    stage_manifest = workspace / "detect_person" / cfg.run_name / "manifest.csv"
    ctx.manifest_path = stage_manifest
    ctx.manifest_producer = "detect_person"

    stats = ctx.stats.get("detect_person", {})
    print(f"  detected={stats.get('detected', 0)}  "
          f"fallback={stats.get('fallback', 0)}  "
          f"errors={stats.get('errors', 0)}")

    # Show bbox results from the stage manifest (via ctx.manifest_df)
    df = ctx.manifest_df
    print("\n  Manifest BBOX columns:")
    for _, row in df.iterrows():
        detected = row.get("PERSON_DETECTED", "N/A")
        sample_id = row.get("SAMPLE_ID", row.get("SENTENCE_NAME", "?"))
        print(f"    {sample_id:20s}  detected={detected}  "
              f"bbox=({row.get('BBOX_X1', 'N/A'):.0f}, {row.get('BBOX_Y1', 'N/A'):.0f}, "
              f"{row.get('BBOX_X2', 'N/A'):.0f}, {row.get('BBOX_Y2', 'N/A'):.0f})")

    # ----------------------------------------------------------------
    # Step 2: clip_video
    # ----------------------------------------------------------------
    print("\n[3/4] Running clip_video...")
    clip_processor = ClipVideoProcessor(cfg)
    ctx = clip_processor.run(ctx)

    # Manual context routing
    ctx.video_dir = clips_dir
    ctx.video_dir_producer = "clip_video"

    clip_stats = ctx.stats.get("clip_video", {})
    print(f"  total={clip_stats.get('total', 0)}  "
          f"success={clip_stats.get('success', 0)}  "
          f"errors={clip_stats.get('errors', 0)}")

    clips_found = list(clips_dir.glob("*.mp4"))
    print(f"  Clips created: {len(clips_found)}")

    # ----------------------------------------------------------------
    # Step 3: crop_video
    # ----------------------------------------------------------------
    print("\n[4/4] Running crop_video...")
    crop_processor = CropVideoProcessor(cfg)
    ctx = crop_processor.run(ctx)

    # Manual context routing
    ctx.video_dir = cropped_dir
    ctx.video_dir_producer = "crop_video"

    crop_stats = ctx.stats.get("crop_video", {})
    print(f"  total={crop_stats.get('total', 0)}  "
          f"cropped={crop_stats.get('cropped', 0)}  "
          f"copied_no_person={crop_stats.get('copied_no_person', 0)}  "
          f"errors={crop_stats.get('errors', 0)}")

    # ----------------------------------------------------------------
    # Summary: show output dimensions
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")

    cropped_files = list(cropped_dir.glob("*.mp4"))
    if not cropped_files:
        print("  [WARNING] No cropped clips produced!")
    else:
        for f in sorted(cropped_files):
            video_capture = cv2.VideoCapture(str(f))
            w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_capture.release()
            print(f"  {f.name:30s}  {w}x{h}  {frames} frames @ {fps:.1f} fps")

    print(f"\n  Output directory: {workspace}")
    print(f"  Clips    : {clips_dir}")
    print(f"  Cropped  : {cropped_dir}")
    print(f"\n  Open the cropped clips to visually verify the person is centred.\n")

    return workspace


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for detect_person + crop_video"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to any .mp4 video file to test with"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for YOLOv8 (default: cuda:0, use 'cpu' if no GPU)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.25,
        help="Crop padding ratio (default: 0.25)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=5,
        help="Max frames to sample for detection (default: 5)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: dataset/smoke_test/ inside the project root)"
    )
    parser.add_argument(
        "--num-segments", type=int, default=2,
        help="Number of segments to sample from the video (default: 2)"
    )
    parser.add_argument(
        "--segment-duration", type=float, default=10.0,
        help="Duration of each segment in seconds (default: 10.0)"
    )
    args = parser.parse_args()

    run_smoke_test(
        video_path=args.video,
        device=args.device,
        padding=args.padding,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        num_segments=args.num_segments,
        segment_duration=args.segment_duration,
    )


if __name__ == "__main__":
    main()
