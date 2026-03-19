"""Pydantic configuration models."""

from pydantic import BaseModel, field_validator
from typing import Any, Dict, Optional, List, Literal


class ExtractorConfig(BaseModel):
    name: str = "mediapipe"
    max_workers: int = 4
    batch_size: int = 16  # Number of frames per batch for inference
    # MediaPipe-specific
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = True
    # MMPose-specific
    pose_model_config: str = ""
    pose_model_checkpoint: str = ""
    det_model_config: str = ""
    det_model_checkpoint: str = ""
    bbox_threshold: float = 0.5
    keypoint_threshold: float = 0.3
    add_visible: bool = True
    device: str = "cuda:0"


class NormalizeConfig(BaseModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = "xy_isotropic_z_minmax"
    remove_z: bool = False
    select_keypoints: bool = True
    keypoint_preset: Optional[str] = None
    keypoint_indices: Optional[List[int]] = None
    mask_empty_frames: bool = True
    mask_low_confidence: bool = False
    visibility_threshold: float = 0.3
    missing_value: float = -999.0

    @field_validator("keypoint_preset")
    @classmethod
    def validate_keypoint_preset(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            from sign_prep.presets import KEYPOINT_PRESETS
            if v not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset '{v}'. "
                    f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
                )
        return v


class ProcessingConfig(BaseModel):
    max_workers: int = 4
    target_fps: Optional[float] = 24.0
    frame_skip: int = 2
    accept_fps_range: Optional[List[float]] = [24.0, 60.0]
    skip_existing: bool = True
    min_duration: float = 0.2
    max_duration: float = 60.0
    signer_policy: Literal["primary_signer", "single_person", "any"] = "primary_signer"


class WebDatasetConfig(BaseModel):
    max_shard_count: int = 10000
    max_shard_size: Optional[int] = None


class ClipVideoConfig(BaseModel):
    codec: str = "copy"
    resize: Optional[List[int]] = None


class DetectPersonConfig(BaseModel):
    enabled: bool = False
    model: str = "yolov8n.pt"
    backend: Literal["ultralytics"] = "ultralytics"
    confidence_threshold: float = 0.5
    sample_strategy: Literal["skip_frame", "uniform"] = "skip_frame"
    uniform_frames: int = 5     # uniform mode: exact number of frames to sample
    max_frames: int = 5         # skip_frame mode: maximum frames to sample
    # frame_skip is read from processing.frame_skip — not duplicated here
    device: str = "cpu"
    min_bbox_area: float = 0.05


class CropVideoConfig(BaseModel):
    enabled: bool = False
    padding: float = 0.25   # Padding ratio around the detected bbox
    codec: str = "libx264"


class PathsConfig(BaseModel):
    root: str = ""
    videos: str = ""
    transcripts: str = ""
    manifest: str = ""
    landmarks: str = ""
    normalized: str = ""
    clips: str = ""
    cropped_clips: str = ""
    webdataset: str = ""


class Config(BaseModel):
    dataset: str
    recipe: Literal["pose", "video"] = "pose"
    run_name: str = "default"
    start_from: Optional[str] = None
    stop_at: Optional[str] = None

    # Source config — only accessed by dataset adapter
    source: Dict[str, Any] = {}

    # Per-stage extension point — keyed by stage name
    stage_config: Dict[str, Dict[str, Any]] = {}

    extractor: ExtractorConfig = ExtractorConfig()
    paths: PathsConfig = PathsConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    processing: ProcessingConfig = ProcessingConfig()
    webdataset: WebDatasetConfig = WebDatasetConfig()
    clip_video: ClipVideoConfig = ClipVideoConfig()
    detect_person: DetectPersonConfig = DetectPersonConfig()
    crop_video: CropVideoConfig = CropVideoConfig()
