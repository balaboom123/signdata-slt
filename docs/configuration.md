# Configuration Reference

## Inheritance & Overrides

Config values are merged in this order (later wins):

1. **Pydantic defaults** -- hardcoded in `config/schema.py`
2. **Base YAML** -- loaded via the `_base` key (e.g. `_base: _base/pose_mediapipe.yaml`)
3. **Dataset YAML** -- the file you pass on the command line
4. **CLI overrides** -- `--override key=value` arguments

```bash
python -m signdata configs/youtube_asl/pose_mediapipe.yaml \
  --override processing.max_workers=8 normalize.remove_z=true
```

### Merge example

`_base/pose_mediapipe.yaml` sets:
```yaml
processing:
  max_workers: 4
  target_fps: 24.0
```

`configs/youtube_asl/pose_mediapipe.yaml` overrides only what differs:
```yaml
_base: _base/pose_mediapipe.yaml

processing:
  max_workers: 8   # overrides the base value; target_fps: 24.0 is inherited unchanged
```

## Minimal Working Config

The smallest valid dataset config only needs to specify what differs from the base and Pydantic defaults:

```yaml
# configs/my_dataset/pose_mediapipe.yaml
_base: _base/pose_mediapipe.yaml

dataset: my_dataset

pipeline:
  steps: [extract, normalize, webdataset]

paths:
  root: dataset/my_dataset
  videos: dataset/my_dataset/videos
  manifest: dataset/my_dataset/manifest.csv
```

Everything else (extractor settings, normalization mode, worker counts, etc.) is inherited from the base YAML and Pydantic defaults.

## Config Files

```
configs/
├── _base/
│   ├── pose_mediapipe.yaml     # Base MediaPipe extractor + normalize + processing
│   ├── pose_mmpose.yaml        # Base MMPose extractor + normalize + processing
│   └── video.yaml              # Base video-mode settings
├── youtube_asl/
│   ├── pose_mediapipe.yaml     # YouTube-ASL + MediaPipe
│   ├── pose_mmpose.yaml        # YouTube-ASL + MMPose
│   └── video.yaml              # YouTube-ASL + video clips
└── how2sign/
    ├── pose_mediapipe.yaml     # How2Sign + MediaPipe
    ├── pose_mmpose.yaml        # How2Sign + MMPose
    └── video.yaml              # How2Sign + video clips
```

## Config Sections

### `pipeline`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `"pose"` \| `"video"` | `"pose"` | Output mode: landmarks or video clips |
| `steps` | `list[str]` | `[]` | Ordered processor names to run |
| `start_from` | `str?` | `null` | Resume from this step (inclusive) |
| `stop_at` | `str?` | `null` | Stop after this step (inclusive) |

### `paths`

All paths are resolved relative to the project root if not absolute. Defaults are derived from `dataset/{dataset_name}/`.

| Field | Type | Default | Description |
|---|---|---|---|
| `root` | `str` | `""` | Dataset root directory |
| `videos` | `str` | `""` | Downloaded video files |
| `transcripts` | `str` | `""` | Transcript JSON files |
| `manifest` | `str` | `""` | Manifest CSV path |
| `landmarks` | `str` | `""` | Raw extracted landmarks |
| `normalized` | `str` | `""` | Normalized landmarks |
| `clips` | `str` | `""` | Clipped video segments |
| `webdataset` | `str` | `""` | Output tar shards |

### `download`

| Field | Type | Default | Description |
|---|---|---|---|
| `video_ids_file` | `str` | `""` | Path to video ID list file |
| `languages` | `list[str]` | `["en"]` | Transcript language codes |
| `format` | `str` | `"worstvideo[height>=720]..."` | yt-dlp format selector |
| `rate_limit` | `str` | `"5M"` | Download rate limit |
| `concurrent_fragments` | `int` | `5` | Parallel download fragments |

### `manifest`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_text_length` | `int` | `300` | Max characters per segment |
| `min_duration` | `float` | `0.2` | Min segment duration (seconds) |
| `max_duration` | `float` | `60.0` | Max segment duration (seconds) |

### `extractor`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"mediapipe"` | Extractor name (`mediapipe` or `mmpose`) |
| `max_workers` | `int` | `4` | Parallel extraction workers |
| **MediaPipe** | | | |
| `model_complexity` | `int` | `1` | Model complexity (0, 1, or 2) |
| `min_detection_confidence` | `float` | `0.5` | Detection confidence threshold |
| `min_tracking_confidence` | `float` | `0.5` | Tracking confidence threshold |
| `refine_face_landmarks` | `bool` | `true` | Use 478 face landmarks (vs 468) |
| **MMPose** | | | |
| `pose_model_config` | `str` | `""` | RTMPose3D model config path |
| `pose_model_checkpoint` | `str` | `""` | RTMPose3D checkpoint path |
| `det_model_config` | `str` | `""` | RTMDet model config path |
| `det_model_checkpoint` | `str` | `""` | RTMDet checkpoint path |
| `bbox_threshold` | `float` | `0.5` | Detection bounding-box threshold |
| `keypoint_threshold` | `float` | `0.3` | Keypoint confidence threshold |
| `add_visible` | `bool` | `true` | Include visibility as 4th channel |
| `device` | `str` | `"cuda:0"` | Inference device |

### `normalize`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"xy_isotropic_z_minmax"` | Normalization mode (`isotropic_3d` or `xy_isotropic_z_minmax`) |
| `remove_z` | `bool` | `false` | Drop z-coordinate after normalization |
| `select_keypoints` | `bool` | `true` | Reduce to subset of keypoints |
| `keypoint_indices` | `list[int]?` | `null` | Custom keypoint indices (auto-detected if null) |
| `mask_empty_frames` | `bool` | `true` | Mask frames with all-zero landmarks |
| `mask_low_confidence` | `bool` | `false` | Mask individual low-visibility landmarks |
| `visibility_threshold` | `float` | `0.3` | Visibility score cutoff |
| `missing_value` | `float` | `-999.0` | Fill value for masked landmarks |

### `processing`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_workers` | `int` | `4` | Parallel processing workers |
| `target_fps` | `float?` | `24.0` | Target frame rate for sampling |
| `frame_skip` | `int` | `2` | Skip every N frames |
| `accept_fps_range` | `list[float]?` | `[24.0, 60.0]` | Acceptable source FPS range |
| `skip_existing` | `bool` | `true` | Skip already-processed files |
| `min_duration` | `float` | `0.2` | Min segment duration (seconds) |
| `max_duration` | `float` | `60.0` | Max segment duration (seconds) |

### `webdataset`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_shard_count` | `int` | `10000` | Max samples per shard |
| `max_shard_size` | `int?` | `null` | Max bytes per shard (optional) |

### `clip_video`

| Field | Type | Default | Description |
|---|---|---|---|
| `codec` | `str` | `"copy"` | ffmpeg codec (`copy` for fast, `libx264` for re-encode) |
| `resize` | `list[int]?` | `null` | Optional `[width, height]` rescale |

## CLI Override Examples

```bash
# Change number of workers
--override processing.max_workers=8

# Switch extractor
--override extractor.name=mmpose

# Disable z-coordinate removal
--override normalize.remove_z=false

# Run only extraction and normalization
--override pipeline.start_from=extract pipeline.stop_at=normalize

# Use a specific GPU
--override extractor.device=cuda:1
```

---

## See Also

- [Architecture](architecture.md) -- system design, registry, pipeline flow
- [Pipeline Stages](pipeline-stages.md) -- what each stage does and its I/O
- [Installation Guide](installation.md) -- base setup and MMPose dependencies
