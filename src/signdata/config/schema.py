"""Pydantic configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Detection configs (one per backend, no `type` field) ---

class YOLODetectionConfig(StrictModel):
    model: str = Field(default="yolo11m.pt", description="YOLO model alias or local weights path.")
    device: str = Field(default="cpu", description="Ultralytics inference device, for example cpu or cuda:0.")
    confidence_threshold: float = Field(default=0.5, description="Minimum person detection confidence.")
    min_bbox_area: float = Field(default=0.05, description="Minimum detection box area as a fraction of frame area.")
    batch_size: int = Field(default=16, gt=0, description="Frames per detector inference batch.")
    allow_download: bool = Field(default=True, description="Allow download of missing official YOLO weights.")
    weights_dir: str | None = Field(default=None, description="Optional directory used to resolve local YOLO weights.")


class MMDetDetectionConfig(StrictModel):
    det_model_config: str = Field(description="MMDetection model config path.")
    det_model_checkpoint: str = Field(description="MMDetection checkpoint path or URL.")
    device: str = Field(default="cuda:0", description="MMDetection inference device.")
    batch_size: int = Field(default=16, gt=0, description="Frames per detector inference batch.")


class MediaPipeDetectionConfig(StrictModel):
    min_detection_confidence: float = Field(default=0.5, description="Minimum MediaPipe person detection confidence.")


# --- Pose configs (one per backend, no `type` field) ---

class MediaPipePoseConfig(StrictModel):
    model_complexity: int = Field(default=1, description="MediaPipe Holistic model complexity.")
    min_detection_confidence: float = Field(default=0.5, description="Minimum initial landmark detection confidence.")
    min_tracking_confidence: float = Field(default=0.5, description="Minimum landmark tracking confidence.")
    refine_face_landmarks: bool = Field(default=True, description="Enable refined face landmarks, producing 553 total points.")
    batch_size: int = Field(default=16, description="Frames passed to the pose backend per pipeline batch.")


class MMPosePoseConfig(StrictModel):
    pose_model_config: str = Field(description="MMPose model config path or package resource.")
    pose_model_checkpoint: str = Field(description="MMPose checkpoint path or URL.")
    device: str = Field(default="cuda:0", description="MMPose inference device.")
    batch_size: int = Field(default=16, description="Frames passed to the pose backend per pipeline batch.")


# --- Config selector maps (used by model_validator) ---

DETECTION_CONFIG_MAP = {
    "yolo": YOLODetectionConfig,
    "mmdet": MMDetDetectionConfig,
    "mediapipe": MediaPipeDetectionConfig,
    "null": None,
}

POSE_CONFIG_MAP = {
    "mediapipe": MediaPipePoseConfig,
    "mmpose": MMPosePoseConfig,
}


# --- Video processing config ---

NVENC_PRESETS = frozenset(f"p{index}" for index in range(1, 8))

# Encoders whose speed knob is an integer rather than an x264-style name, as
# (ffmpeg flag, low, high) over the inclusive range ffmpeg accepts.
#
# The flag travels with the range because it is not always -preset: libaom-av1
# and libvpx-vp9 expose no -preset at all and use -cpu-used instead, so
# `_encoder_args` emits whichever flag the codec actually has. All three count
# the opposite way to x264 — the *high* end is the fastest.
NUMERIC_PRESET_SPECS = {
    "libsvtav1": ("-preset", -2, 13),
    "libaom-av1": ("-cpu-used", 0, 8),
    "libvpx-vp9": ("-cpu-used", -8, 8),
}


class VideoProcessingConfig(StrictModel):
    codec: str = Field(default="libx264", description="FFmpeg encoder used for crop or compression output.")
    padding: float = Field(default=0.0, description="Fractional padding added around a detected crop box.")
    resize: list[int] | None = Field(default=None, description="Optional output size as [width, height]; compression forbids resizing.")
    # Constant-quality level: -crf on software encoders, -cq on NVENC. AV1
    # allows up to 63; h264/hevc cap at 51.
    crf: int = Field(default=20, ge=0, le=63, description="Constant-quality value used by video compression; crop uses 15.")
    preset: str = Field(default="medium", description="Encoder speed/efficiency preset in codec-specific form.")
    # NVENC spatial/temporal AQ strength (1 weakest, 15 strongest).
    aq_strength: int = Field(default=8, ge=1, le=15, description="NVENC spatial and temporal adaptive-quantization strength.")
    # Physical NVIDIA GPU index passed to NVENC/NVDEC; null uses FFmpeg default.
    nvenc_gpu: int | None = Field(default=None, ge=0, description="Physical NVIDIA GPU index; null lets FFmpeg choose.")
    # VBV ceiling as a fraction of the source bitrate; null disables the cap.
    max_bitrate_ratio: float | None = Field(default=0.8, gt=0.0, description="Maximum output bitrate as a fraction of source bitrate.")
    min_video_reduction_ratio: float = Field(default=0.10, ge=0.0, lt=1.0, description="Minimum size reduction required to keep a re-encode.")
    encoding_timeout_seconds: int = Field(default=3600, gt=0, description="Per-video FFmpeg encoding timeout in seconds.")

    @field_validator("preset", mode="before")
    @classmethod
    def coerce_numeric_preset(cls, value):
        """Accept `preset: 8` unquoted, the natural way to write libsvtav1.

        Pydantic will not narrow an int to a str, so without this the correct
        value for a numeric-preset encoder fails with a generic type error
        instead of the codec-aware message below.
        """
        return str(value) if isinstance(value, int) else value

    @model_validator(mode="after")
    def validate_codec_preset(self):
        """Catch a codec/preset mismatch at config load, not 3 hours in.

        Each encoder family names its presets differently, and ffmpeg only
        complains once the encode starts. Checked in family order so the
        error message can suggest the form that codec actually wants.
        """
        if self.codec.endswith("_nvenc"):
            if self.preset not in NVENC_PRESETS:
                raise ValueError(
                    f"codec={self.codec!r} needs an NVENC preset p1-p7 "
                    f"(p1 fastest, p7 best), got {self.preset!r}"
                )
            return self

        numeric_spec = NUMERIC_PRESET_SPECS.get(self.codec)
        if numeric_spec is not None:
            flag, low, high = numeric_spec
            wanted = (
                f"codec={self.codec!r} takes a numeric preset {low} to {high} "
                f"({high} fastest, sent to ffmpeg as {flag})"
            )
            try:
                level = int(self.preset)
            except ValueError:
                raise ValueError(
                    f"{wanted}, not an x264-style name like {self.preset!r}"
                ) from None
            if not low <= level <= high:
                raise ValueError(f"{wanted}, got {level}")
            return self

        if self.preset in NVENC_PRESETS:
            raise ValueError(
                f"preset={self.preset!r} is NVENC-only, but codec="
                f"{self.codec!r} is a software encoder (try 'medium')"
            )
        return self


class PartsProcessingConfig(StrictModel):
    crop_size: int = Field(default=224, gt=0, description="Square size in pixels for face and hand streams.")
    bbox_scale: float = Field(default=1.2, gt=0, description="Scale applied to face and hand landmark boxes.")
    codec: str = Field(default="mp4v", description="OpenCV FourCC used for part-stream MP4 files.")
    missing_value: float = Field(default=-1.0, description="Sentinel for missing body pose values and boxes.")


# --- Stage configs ---

class DatasetConfig(StrictModel):
    name: str = Field(description="Dataset name registered in DATASET_REGISTRY.")
    download: bool = Field(default=True, description="Run the dataset download or local-validation stage.")
    manifest: bool = Field(default=True, description="Build a manifest; false loads paths.manifest when set.")
    source: dict[str, Any] = Field(default_factory=dict, description="Adapter-specific source settings.")


class ProcessingConfig(StrictModel):
    enabled: bool = Field(default=True, description="Run the selected processing stage.")
    processor: Literal[
        "video2pose",
        "video2crop",
        "video2compression",
        "video2parts",
    ] = Field(default="video2pose", description="Main processor implementation.")
    detection: Literal["yolo", "mediapipe", "mmdet", "null"] = Field(default="null", description="Person detection backend.")
    pose: Literal["mediapipe", "mmpose"] | None = Field(default=None, description="Pose backend required by video2pose and video2parts.")

    # Shared sampling param: native / ratio / absolute FPS
    sample_rate: float | None = Field(default=0.5, description="Frame sampling mode: null native, fraction ratio, or absolute FPS.")
    max_workers: int = Field(default=1, gt=0, description="Worker count used by post-processing and compression.")

    # Backend-specific configs — raw dicts from YAML, validated into typed
    # models by model_validator and stored back as public attributes.
    detection_config: (
        YOLODetectionConfig
        | MMDetDetectionConfig
        | MediaPipeDetectionConfig
        | dict
        | None
    ) = Field(default=None, description="Configuration for the selected detection backend.")
    pose_config: MediaPipePoseConfig | MMPosePoseConfig | dict | None = Field(default=None, description="Configuration for the selected pose backend.")
    video_config: VideoProcessingConfig | None = Field(default=None, description="Video crop or compression settings.")
    parts_config: PartsProcessingConfig | dict | None = Field(default=None, description="Face, hand, and body-stream settings for video2parts.")

    @model_validator(mode="before")
    @classmethod
    def reject_removed_sampling_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            removed = {"frame_skip", "target_fps"}.intersection(data)
            if removed:
                raise ValueError(
                    f"Removed processing option(s): {', '.join(sorted(removed))}; "
                    "use processing.sample_rate"
                )
        return data

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("sample_rate must be positive or null")
        return v

    @model_validator(mode="after")
    def validate_backend_configs(self):
        """Parse raw dicts into typed config models."""
        # Skip validation when processing is disabled (allows default construction)
        if not self.enabled:
            return self

        # Validate + replace detection_config
        det_cls = DETECTION_CONFIG_MAP.get(self.detection)
        if det_cls and self.detection_config is not None:
            if isinstance(self.detection_config, dict):
                self.detection_config = det_cls(**self.detection_config)
        elif det_cls and self.detection_config is None:
            raise ValueError(
                f"detection={self.detection!r} requires detection_config"
            )

        # video2pose/video2parts require pose + pose_config
        if self.processor in ("video2pose", "video2parts"):
            if not self.pose:
                raise ValueError(f"processing.pose is required for {self.processor}")
            if self.processor == "video2parts" and self.pose != "mediapipe":
                raise ValueError("video2parts currently requires pose=mediapipe")
            pose_cls = POSE_CONFIG_MAP.get(self.pose)
            if pose_cls and self.pose_config is not None:
                if isinstance(self.pose_config, dict):
                    self.pose_config = pose_cls(**self.pose_config)
            elif pose_cls and self.pose_config is None:
                raise ValueError(
                    f"pose={self.pose!r} requires pose_config"
                )

        if self.processor == "video2parts":
            if isinstance(self.parts_config, dict):
                self.parts_config = PartsProcessingConfig(**self.parts_config)
            elif self.parts_config is None:
                self.parts_config = PartsProcessingConfig()

        # video2crop / video2compression require video_config (defaults if omitted)
        if self.processor in ("video2crop", "video2compression") and not self.video_config:
            self.video_config = VideoProcessingConfig()

        return self


class NormalizeConfig(StrictModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = Field(default="xy_isotropic_z_minmax", description="Coordinate normalization algorithm.")
    remove_z: bool = Field(default=False, description="Drop the z coordinate after normalization.")
    select_keypoints: bool = Field(default=True, description="Apply a preset or explicit keypoint selection before flattening.")
    keypoint_preset: str | None = Field(default=None, description="Named built-in keypoint selection preset.")
    keypoint_indices: list[int] | None = Field(default=None, description="Explicit keypoint indices used when no preset is selected.")
    mask_empty_frames: bool = Field(default=True, description="Replace all-zero frames with the missing-value sentinel.")
    mask_low_confidence: bool = Field(default=False, description="Mask landmarks below the visibility threshold.")
    visibility_threshold: float = Field(default=0.3, description="Visibility cutoff used by low-confidence masking.")
    missing_value: float = Field(default=-999.0, description="Sentinel written for masked coordinates.")

    @field_validator("keypoint_preset")
    @classmethod
    def validate_keypoint_preset(cls, v: str | None) -> str | None:
        if v is not None:
            from signdata.processors.pose import KEYPOINT_PRESETS
            if v not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset '{v}'. "
                    f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
                )
        return v


class PostProcessingConfig(StrictModel):
    enabled: bool = Field(default=False, description="Run landmark post-processing.")
    normalize: NormalizeConfig | None = Field(default=None, description="Landmark normalization settings.")


class OutputConfig(StrictModel):
    enabled: bool = Field(default=True, description="Run WebDataset packaging.")
    max_shard_count: int = Field(default=10_000, gt=0, description="Maximum samples per tar shard.")
    max_shard_size: int | None = Field(default=None, gt=0, description="Approximate maximum payload bytes per shard.")


class PathsConfig(StrictModel):
    root: str = Field(default="", description="Dataset root used to derive omitted paths.")
    videos: str = Field(default="", description="Source video directory.")
    transcripts: str = Field(default="", description="Transcript JSON directory.")
    manifest: str = Field(default="", description="Manifest TSV path.")
    output: str = Field(default="", description="Base processor artifact directory.")
    webdataset: str = Field(default="", description="Base WebDataset shard directory.")


# --- Top-level ---

class Config(StrictModel):
    dataset: DatasetConfig = Field(description="Dataset adapter settings.")
    processing: ProcessingConfig = Field(
        default_factory=lambda: ProcessingConfig(enabled=False),
        description="Main processor selection and backend settings.",
    )
    post_processing: PostProcessingConfig = Field(
        default_factory=PostProcessingConfig,
        description="Optional landmark post-processing settings.",
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="WebDataset output settings.")
    run_name: str = Field(default="default", description="Namespace appended to output and webdataset paths.")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Resolved dataset and artifact paths.")
