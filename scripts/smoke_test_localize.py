"""Smoke test for person_localize and crop_video processors.

Usage:
    python scripts/smoke_test_localize.py --video path/to/any_video.mp4

This script:
  1. Creates a temporary workspace with a minimal manifest
  2. Runs PersonLocalizeProcessor  → writes BBOX_* columns to manifest
  3. Runs ClipVideoProcessor       → clips the segment
  4. Runs CropVideoProcessor       → crops the clip to the detected person
  5. Prints a summary and shows where to find the output files

No dataset download required — just supply any .mp4 file.
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import pandas as pd

# Make sure src/ is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0.0


def build_manifest(video_path: str, workspace: Path) -> Path:
    """Create a minimal manifest with 2 segments from the video."""
    duration = get_video_duration(video_path)
    if duration < 2.0:
        raise ValueError(f"Video too short ({duration:.1f}s), need at least 2s.")

    video_name = Path(video_path).stem

    # Create 2 segments: first half and second half
    mid = duration / 2
    seg1_end = min(mid, 10.0)       # cap at 10s
    seg2_start = mid
    seg2_end = min(duration, mid + 10.0)

    manifest_path = workspace / "manifest.csv"
    df = pd.DataFrame({
        "VIDEO_NAME":       [video_name,         video_name],
        "SENTENCE_NAME":    [f"{video_name}-0",  f"{video_name}-1"],
        "START_REALIGNED":  [0.0,                seg2_start],
        "END_REALIGNED":    [seg1_end,            seg2_end],
        "SENTENCE":         ["segment 0",         "segment 1"],
    })
    df.to_csv(manifest_path, sep="\t", index=False)
    print(f"  Created manifest: {manifest_path}")
    print(f"  Segment 0: 0.0s → {seg1_end:.1f}s")
    print(f"  Segment 1: {seg2_start:.1f}s → {seg2_end:.1f}s")
    return manifest_path


def run_smoke_test(video_path: str, device: str, padding: float, sample_frames: int):
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
    from sign_prep.processors.common.person_localize import PersonLocalizeProcessor
    from sign_prep.processors.common.clip_video import ClipVideoProcessor
    from sign_prep.processors.common.crop_video import CropVideoProcessor

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Smoke test: person_localize + clip_video + crop_video")
    print(f"  Video : {video_path}")
    print(f"  Device: {device}  |  Padding: {padding}  |  Sample frames: {sample_frames}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # Setup workspace
    # ----------------------------------------------------------------
    workspace = Path(tempfile.mkdtemp(prefix="sign_prep_smoke_"))
    videos_dir  = workspace / "videos"
    clips_dir   = workspace / "clips"
    cropped_dir = workspace / "cropped_clips"
    videos_dir.mkdir()
    clips_dir.mkdir()
    cropped_dir.mkdir()

    # Symlink (or copy) the video into the videos dir so the processor finds it
    video_name = Path(video_path).stem
    dest_video = videos_dir / f"{video_name}.mp4"
    try:
        os.symlink(video_path, dest_video)
    except (OSError, NotImplementedError):
        # Windows may not support symlinks without admin rights → copy instead
        shutil.copy2(video_path, dest_video)

    print(f"[1/4] Building manifest...")
    manifest_path = build_manifest(str(dest_video), workspace)

    # ----------------------------------------------------------------
    # Build config
    # ----------------------------------------------------------------
    cfg = Config(
        dataset="youtube_asl",
        pipeline={"mode": "video", "steps": ["person_localize", "clip_video", "crop_video"]},
        paths={
            "root":          str(workspace),
            "videos":        str(videos_dir),
            "manifest":      str(manifest_path),
            "clips":         str(clips_dir),
            "cropped_clips": str(cropped_dir),
        },
        person_localize={
            "model":                "yolov8n.pt",
            "confidence_threshold": 0.5,
            "sample_frames":        sample_frames,
            "device":               device,
            "min_bbox_area":        0.02,   # relaxed for smoke test
        },
        crop_video={
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
    )

    # ----------------------------------------------------------------
    # Step 1: person_localize
    # ----------------------------------------------------------------
    print("\n[2/4] Running person_localize...")
    processor = PersonLocalizeProcessor(cfg)
    ctx = processor.run(ctx)

    stats = ctx.stats.get("person_localize", {})
    print(f"  detected={stats.get('detected', 0)}  "
          f"fallback={stats.get('fallback', 0)}  "
          f"errors={stats.get('errors', 0)}")

    # Show bbox results
    df = pd.read_csv(manifest_path, sep="\t")
    print("\n  Manifest BBOX columns:")
    for _, row in df.iterrows():
        detected = row.get("PERSON_DETECTED", "N/A")
        print(f"    {row['SENTENCE_NAME']:20s}  detected={detected}  "
              f"bbox=({row.get('BBOX_X1', 'N/A'):.0f}, {row.get('BBOX_Y1', 'N/A'):.0f}, "
              f"{row.get('BBOX_X2', 'N/A'):.0f}, {row.get('BBOX_Y2', 'N/A'):.0f})")

    # ----------------------------------------------------------------
    # Step 2: clip_video
    # ----------------------------------------------------------------
    print("\n[3/4] Running clip_video...")
    clip_processor = ClipVideoProcessor(cfg)
    ctx = clip_processor.run(ctx)

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
            cap = cv2.VideoCapture(str(f))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"  {f.name:30s}  {w}x{h}  {frames} frames @ {fps:.1f} fps")

    print(f"\n  Output directory: {workspace}")
    print(f"  Clips    : {clips_dir}")
    print(f"  Cropped  : {cropped_dir}")
    print(f"\n  Open the cropped clips to visually verify the person is centred.\n")

    return workspace


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for person_localize + crop_video"
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
        "--sample-frames", type=int, default=5,
        help="Number of frames to sample for detection (default: 5)"
    )
    args = parser.parse_args()

    run_smoke_test(
        video_path=args.video,
        device=args.device,
        padding=args.padding,
        sample_frames=args.sample_frames,
    )


if __name__ == "__main__":
    main()