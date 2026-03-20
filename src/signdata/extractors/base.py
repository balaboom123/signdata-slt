"""Base class for landmark extractors."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class LandmarkExtractor(ABC):
    """Abstract base class for landmark extraction."""

    # Whether the extractor supports true batch inference (GPU batching)
    supports_batch_inference: bool = False

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame and extract landmarks.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            Numpy array of landmarks, or None if extraction fails.
        """
        pass

    def process_batch(
        self,
        frames: List[np.ndarray],
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of frames and extract landmarks.

        Default implementation processes frames sequentially.
        Subclasses may override for true batch inference.

        Args:
            frames: List of input video frames (BGR format from OpenCV)
            fallback_on_error: If True, continue processing remaining frames
                when one fails. If False, raise the exception.

        Returns:
            List of numpy arrays of landmarks (or None for failed frames).
        """
        results = []
        for frame in frames:
            try:
                landmarks = self.process_frame(frame)
                results.append(landmarks)
            except Exception:
                if fallback_on_error:
                    results.append(None)
                else:
                    raise
        return results

    @abstractmethod
    def close(self):
        """Release resources (models, GPU memory, etc.)."""
        pass
