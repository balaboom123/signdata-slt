"""Video processing utilities."""

import os
from typing import Optional

import cv2


def validate_video_file(video_path: str) -> bool:
    """Validate if a video file exists and can be opened by OpenCV."""
    if not os.path.exists(video_path):
        return False
    try:
        video_capture = cv2.VideoCapture(video_path)
        is_valid = video_capture.isOpened()
        video_capture.release()
        return is_valid
    except Exception:
        return False


def get_video_fps(video_path: str) -> float:
    """Return video FPS (frames per second) as float."""
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return 0.0
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
        video_capture.release()
        return float(fps)
    except Exception:
        return 0.0


class FPSSampler:
    """Frame sampling strategies for video processing.

    Supports two sampling modes:
      1) reduce mode (priority): Downsample source fps to target fps
         Uses accumulation error method (Bresenham-like) for non-integer ratios
      2) skip mode: Sample every Nth frame
    """

    def __init__(self, src_fps: float, reduce_to: Optional[float], frame_skip_by: int):
        self.mode = "reduce" if (reduce_to is not None and src_fps > 0) else "skip"

        if self.mode == "reduce":
            self.target = min(reduce_to, src_fps)
            self.r = self.target / max(src_fps, 1e-6)
            self.acc = 0.0
        else:
            self.n = max(int(frame_skip_by), 1)
            self.count = 0

    def take(self) -> bool:
        """Returns True if current frame should be sampled."""
        if self.mode == "reduce":
            self.acc += self.r
            if self.acc >= 1.0:
                self.acc -= 1.0
                return True
            return False
        else:
            take_now = (self.count % self.n) == 0
            self.count += 1
            return take_now
