"""Pydantic configuration models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class DownloadConfig(BaseModel):
    video_ids_file: str = ""
    languages: List[str] = ["en"]
    format: str = "worstvideo[height>=720][fps>=24]/bestvideo[height>=480]"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5


class ManifestConfig(BaseModel):
    max_text_length: int = 300
    min_duration: float = 0.2
    max_duration: float = 60.0


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
    kpt_threshold: float = 0.3
    add_visible: bool = True
    device: str = "cuda:0"


class NormalizeConfig(BaseModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = "xy_isotropic_z_minmax"
    remove_z: bool = False
    reduction: bool = True
    keypoint_indices: Optional[List[int]] = None
    mask_frame_level: bool = True
    mask_landmark_level: bool = False
    visibility_threshold: float = 0.3
    unvisible_value: float = -999.0


class ProcessingConfig(BaseModel):
    max_workers: int = 4
    target_fps: Optional[float] = 24.0
    frame_skip: int = 2
    accept_fps_range: Optional[List[float]] = [24.0, 60.0]
    skip_existing: bool = True
    min_duration: float = 0.2
    max_duration: float = 60.0


class WebDatasetConfig(BaseModel):
    max_shard_count: int = 10000
    max_shard_size: Optional[int] = None


class ClipVideoConfig(BaseModel):
    codec: str = "copy"
    resize: Optional[List[int]] = None


class PersonLocalizeConfig(BaseModel):
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
    padding: float = 0.25   # Padding ratio around the detected bbox
    codec: str = "libx264"


class PipelineConfig(BaseModel):
    mode: Literal["pose", "video"] = "pose"
    steps: List[str] = []
    start_from: Optional[str] = None
    stop_at: Optional[str] = None


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
    pipeline: PipelineConfig = PipelineConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    paths: PathsConfig = PathsConfig()
    download: DownloadConfig = DownloadConfig()
    manifest: ManifestConfig = ManifestConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    processing: ProcessingConfig = ProcessingConfig()
    webdataset: WebDatasetConfig = WebDatasetConfig()
    clip_video: ClipVideoConfig = ClipVideoConfig()
    person_localize: PersonLocalizeConfig = PersonLocalizeConfig()
    crop_video: CropVideoConfig = CropVideoConfig()