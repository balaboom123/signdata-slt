"""MMPose-based 3D pose landmark extraction."""

from typing import Optional, List

import numpy as np

from .base import LandmarkExtractor
from ..registry import register_extractor
from ..config.schema import ExtractorConfig


class MultiPersonDetected(Exception):
    """Raised when more than one person is detected in a frame."""
    pass


@register_extractor("mmpose")
class MMPoseExtractor(LandmarkExtractor):
    """Extracts 3D pose landmarks using MMPose RTMPose3D.

    Uses a two-stage pipeline:
    1. RTMDet for person detection (bounding boxes)
    2. RTMPose3D for 3D pose estimation

    Always outputs all 133 COCO WholeBody keypoints.
    """

    def __init__(
        self,
        config: ExtractorConfig,
        detector=None,
        pose_estimator=None,
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.bbox_threshold = config.bbox_threshold
        self.det_cat_id = 0
        self.add_visible = config.add_visible

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect person and extract 3D landmarks from a single frame.

        Returns array of shape (133, 4) with [x, y, z, visibility].
        Raises MultiPersonDetected if multiple persons detected.
        """
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        from mmdet.apis import inference_detector

        det_result = inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(
            pred_instance.labels == self.det_cat_id,
            pred_instance.scores > self.bbox_threshold
        )]

        if len(bboxes) == 0:
            return None

        if len(bboxes) > 1:
            raise MultiPersonDetected(
                f"Detected {len(bboxes)} persons with score > {self.bbox_threshold}"
            )

        pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)

        for idx, pose_est_result in enumerate(pose_est_results):
            pose_est_result.track_id = pose_est_results[idx].get("track_id", 1e4)

            pred_instances = pose_est_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores

            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores

            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            pose_est_results[idx].pred_instances.keypoints = keypoints

        pose_est_results = sorted(
            pose_est_results, key=lambda x: x.get("track_id", 1e4)
        )

        pred_3d_data_samples = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get("pred_instances", None)

        if pred_3d_instances is None:
            return None

        H, W = frame.shape[:2]
        packed = self._pack_keypoints(pred_3d_instances, W, H)
        return packed

    def _pack_keypoints(
        self,
        pred_3d_instances,
        img_w: int,
        img_h: int,
        instance_index: int = 0,
    ) -> Optional[np.ndarray]:
        """Extract and pack keypoints from 3D pose estimation results."""
        if pred_3d_instances is None:
            return None

        tk = getattr(pred_3d_instances, "transformed_keypoints", None)
        k3d = getattr(pred_3d_instances, "keypoints", None)
        if tk is None or k3d is None:
            return None

        tk = self._to_numpy(tk)
        k3d = self._to_numpy(k3d)
        tk = self._squeeze_kpts(tk)
        k3d = self._squeeze_kpts(k3d)

        if tk.ndim != 3 or k3d.ndim != 3 or tk.shape[0] == 0 or k3d.shape[0] == 0:
            return None

        xy = tk[instance_index]
        xyz = k3d[instance_index]

        x_norm = xy[..., 0] / float(img_w)
        y_norm = xy[..., 1] / float(img_h)
        z = xyz[..., 2]

        kpt_scores = getattr(pred_3d_instances, "keypoint_scores", None)
        if kpt_scores is not None:
            kpt_scores = self._to_numpy(kpt_scores)
            if kpt_scores.ndim == 2:
                visible = kpt_scores[instance_index]
            elif kpt_scores.ndim == 3:
                visible = kpt_scores[instance_index, :, 0]
            else:
                visible = np.ones(133, dtype=np.float32)
        else:
            visible = np.ones(133, dtype=np.float32)

        out = np.stack([x_norm, y_norm, z, visible], axis=-1).astype(np.float32)
        return out

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _squeeze_kpts(arr, expect_last: int = 2):
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    def close(self):
        pass
