"""MediaPipe-based holistic landmark extraction."""

from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np

from .base import LandmarkExtractor
from ..registry import register_extractor
from ..config.schema import ExtractorConfig


@register_extractor("mediapipe")
class MediaPipeExtractor(LandmarkExtractor):
    """Extracts holistic landmarks using MediaPipe.

    Always outputs all landmarks: 33 pose + 478 face (refined) or
    468 face (unrefined) + 21 left hand + 21 right hand.
    """

    # MediaPipe does not support true batch inference
    supports_batch_inference = False

    def __init__(self, config: ExtractorConfig):
        self.refine_face = config.refine_face_landmarks
        self.face_count = 478 if self.refine_face else 468
        # Total landmark count for convenience
        self.num_landmarks = 33 + self.face_count + 21 + 21

        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity,
            refine_face_landmarks=config.refine_face_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract holistic landmarks from a single frame.

        Returns array of shape (num_keypoints, 4) with [x, y, z, visibility].
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        pose_landmarks = self._convert_all_landmarks_to_array(
            getattr(results.pose_landmarks, "landmark", None), 33
        )
        face_landmarks = self._convert_all_landmarks_to_array(
            getattr(results.face_landmarks, "landmark", None), self.face_count
        )
        left_hand_landmarks = self._convert_all_landmarks_to_array(
            getattr(results.left_hand_landmarks, "landmark", None), 21
        )
        right_hand_landmarks = self._convert_all_landmarks_to_array(
            getattr(results.right_hand_landmarks, "landmark", None), 21
        )

        landmark_array = np.concatenate([
            pose_landmarks,
            face_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        ], axis=0)

        return landmark_array.astype(np.float32)

    def _convert_all_landmarks_to_array(
        self,
        landmarks: Optional[List],
        expected_count: int,
    ) -> np.ndarray:
        """Convert all MediaPipe landmarks to numpy array."""
        if landmarks:
            out = []
            for lm in landmarks:
                vis = getattr(lm, "visibility", 1.0)
                out.append([lm.x, lm.y, lm.z, vis])
            return np.array(out, dtype=np.float32)
        else:
            return np.zeros((expected_count, 4), dtype=np.float32)

    def process_batch(
        self,
        frames: List[np.ndarray],
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of frames sequentially.

        MediaPipe does not support native batch inference, so this
        processes frames one-by-one with per-frame error handling.

        Args:
            frames: List of input video frames (BGR format)
            fallback_on_error: If True, return None for failed frames
                and continue. If False, raise exceptions.

        Returns:
            List of landmark arrays (or None for failed frames).
        """
        # Pre-allocate results list
        results: List[Optional[np.ndarray]] = [None] * len(frames)

        for i, frame in enumerate(frames):
            try:
                landmarks = self.process_frame(frame)
                results[i] = landmarks
            except Exception:
                if not fallback_on_error:
                    raise
                # On error, result stays None

        return results

    def close(self):
        if self.holistic is not None:
            self.holistic.close()
