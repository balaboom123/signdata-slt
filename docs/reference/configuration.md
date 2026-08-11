# Configuration Reference

## Job and Experiment Files

Runnable job YAMLs live under `configs/jobs/<dataset>/`.
Experiment YAMLs live under `configs/experiments/` and reference job paths relative to `configs/`.
Jobs may also declare a top-level `base:` key to inherit one or more shared YAML files before local values are applied.

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8 post_processing.normalize.remove_z=true

python -m signdata experiment configs/experiments/baseline_youtube_asl.yaml
```

## Merge Order

Config values are applied in this order, with later values winning:

1. Pydantic defaults from `src/signdata/config/schema.py`
2. Base YAML values referenced by top-level `base:`
3. Job YAML values
4. CLI overrides passed to `python -m signdata run`
5. Per-job `overrides` from an experiment YAML

## Minimal Working Config

The smallest practical job config needs the fields required by the
dataset adapter and the selected processor:

```yaml
# configs/jobs/how2sign/video.yaml
dataset:
  name: how2sign
  download: true
  manifest: true
  source:
    manifest_csv: dataset/how2sign/text/how2sign_realigned_val.csv

processing:
  enabled: true
  processor: video2crop
  detection: yolo
  detection_config:
    model: yolo11m.pt

run_name: default

paths:
  root: dataset/how2sign
  videos: dataset/how2sign/videos
```

The loader will derive `paths.transcripts`, `paths.manifest`, `paths.output`,
and `paths.webdataset` from `paths.root` when they are omitted.

## Base Config Inheritance

Reusable fragments live under `configs/base/`; runnable jobs live under
`configs/jobs/`. The authoritative package layout is in
[Architecture](../concepts/architecture.md#package-layout), so this page does
not mirror the repository tree.

A job can inherit one base or an ordered list:

```yaml
base: ../../base/datasets/youtube_asl.yaml
```

```yaml
base:
  - ../../base/datasets/youtube_asl.yaml
  - ../../base/post_processing/normalize.yaml
```

Each reference resolves relative to the YAML file that contains it. Bases may
inherit recursively. List entries merge from first to last, then the current
file is merged on top; nested mappings merge recursively and later values win.
Circular references raise an error that reports the full inheritance chain.

## Top-Level Fields

<!-- BEGIN GENERATED: config -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `dataset` | `DatasetConfig` | `required` | — | Dataset adapter settings. |
| `processing` | `ProcessingConfig` | `ProcessingConfig(enabled=false)` | — | Main processor selection and backend settings. |
| `post_processing` | `PostProcessingConfig` | `PostProcessingConfig()` | — | Optional landmark post-processing settings. |
| `output` | `OutputConfig` | `OutputConfig()` | — | WebDataset output settings. |
| `run_name` | `str` | `"default"` | — | Namespace appended to output and webdataset paths. |
| `paths` | `PathsConfig` | `PathsConfig()` | — | Resolved dataset and artifact paths. |
<!-- END GENERATED: config -->

## `paths`

All relative paths are resolved from the project root.

<!-- BEGIN GENERATED: paths -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `root` | `str` | `""` | — | Dataset root used to derive omitted paths. |
| `videos` | `str` | `""` | — | Source video directory. |
| `transcripts` | `str` | `""` | — | Transcript JSON directory. |
| `manifest` | `str` | `""` | — | Manifest TSV path. |
| `output` | `str` | `""` | — | Base processor artifact directory. |
| `webdataset` | `str` | `""` | — | Base WebDataset shard directory. |
<!-- END GENERATED: paths -->

If omitted, the loader derives these defaults from `paths.root`:

- `paths.videos` → `{root}/videos`
- `paths.transcripts` → `{root}/transcripts`
- `paths.manifest` → `{root}/manifest.csv`
- `paths.output` → `{root}/output`
- `paths.webdataset` → `{root}/webdataset`

## `dataset`

<!-- BEGIN GENERATED: dataset -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `name` | `str` | `required` | — | Dataset name registered in DATASET_REGISTRY. |
| `download` | `bool` | `true` | — | Run the dataset download or local-validation stage. |
| `manifest` | `bool` | `true` | — | Build a manifest; false loads paths.manifest when set. |
| `source` | `dict[str, Any]` | `{}` | — | Adapter-specific source settings. |
<!-- END GENERATED: dataset -->

## `dataset.source`

`dataset.source` is adapter-specific. Common keys used by the built-in datasets are:

| Field | Type | Default | Description |
|---|---|---|---|
| `video_ids_file` | `str` | `""` | Video ID list for YouTube-style datasets |
| `annotations_dir` | `str` | `""` | Directory containing MS-ASL JSON annotation files |
| `metadata_file` | `str` | `""` | Optional explicit BOBSL metadata JSON path when `subset2episode.json` is not stored in the default release location |
| `metadata_json` | `str` | `""` | WLASL metadata JSON path |
| `annotation_root` | `str` | `""` | Optional explicit BOBSL isolated-sign annotation path or directory |
| `subtitles_root` | `str` | `""` | Optional explicit BOBSL subtitle directory |
| `release_dir` | `str` | `""` | Local dataset release root for manually downloaded datasets such as AUTSL, LSA64, CSL, SLoVo, and RWTH-PHOENIX-Weather |
| `variant` | `str` | `"cut"` | LSA64 release variant (`cut` or `raw`) |
| `modality` | `str` | adapter defaults | Input modality selector for datasets with paired files such as AUTSL (`rgb` or `depth`) |
| `languages` | `list[str]` | adapter defaults | Transcript language codes |
| `availability_policy` | `str` | `"drop_unavailable"` | Availability handling policy for datasets that may have missing clips |
| `download_mode` | `str` | adapter defaults | Dataset acquisition mode such as `validate` or `download_missing`; WLASL uses `validate` for local preprocessed clips and `download_missing` for raw-source URL fetches |
| `prepare_mode` | `str` | adapter defaults | Local release preparation mode for datasets such as RWTH-PHOENIX-Weather and CSL; CSL uses `validate`, `materialize_missing`, or `rematerialize_all` to select between existing RGB videos and frame-folder materialization |
| `video_fps` | `float` | adapter defaults | Frame rate used when materializing frame-folder datasets such as CSL or RWTH-PHOENIX-Weather into `.mp4` clips |
| `download_format` | `str` | `"worstvideo[...]+worstaudio/.../best"` | yt-dlp format selector |
| `rate_limit` | `str` | `"5M"` | Download rate limit |
| `concurrent_fragments` | `int` | `5` | Parallel download fragments |
| `sleep_interval_requests` | `float` | `0.0` | Delay in seconds between yt-dlp extractor requests; raise this when a video host throttles acquisition |
| `sleep_interval` | `float` | `0.0` | Minimum delay in seconds before each media download |
| `max_sleep_interval` | `float \| null` | `null` | Optional upper bound for randomized per-download delay; must be at least `sleep_interval` |
| `stop_on_download_block` | `bool` | `true` | Stop resumable acquisition promptly when a video host reports HTTP 429 or bot verification instead of treating every remaining video as unavailable |
| `transcript_proxy_http` | `str` | `null` | Optional HTTP proxy for `youtube-transcript-api` |
| `transcript_proxy_https` | `str` | `null` | Optional HTTPS proxy for `youtube-transcript-api` |
| `stop_on_transcript_block` | `bool` | `true` | Stop transcript downloading after `RequestBlocked`/`IpBlocked` |
| `max_text_length` | `int` | `300` | Max caption length |
| `min_duration` | `float` | `0.2` | Min segment duration |
| `max_duration` | `float` | `60.0` | Max segment duration |
| `manifest_csv` | `str` | `""` | Existing manifest path for datasets such as How2Sign |
| `manifest_tsv` | `str` | `""` | Official TSV manifest path for datasets such as OpenASL |
| `annotations_csv` | `str` | `""` | Official annotation CSV path for datasets such as SLoVo |
| `text_column` | `str` | `"raw-text"` | Source text column selected by OpenASL |
| `bbox_json` | `str` | `""` | Optional bounding-box JSON path for datasets such as OpenASL |
| `split` | `str` | `"all"` | Split label for datasets such as WLASL, MS-ASL, and AUTSL |
| `allow_unlabeled` | `bool` | `false` | Include AUTSL videos when split label files are unavailable |
| `view` | `str` | adapter defaults | Dataset view selector for multi-view releases such as BOBSL (`subtitle_slt` or `isolated_signs`) |
| `subtitle_alignment` | `str` | adapter defaults | Subtitle alignment choice for BOBSL (`manual` or `original`) |
| `split_strategy` | `str` | adapter defaults | Split assignment policy for datasets such as LSA64 |
| `protocol` | `str` | adapter defaults | Evaluation protocol name for datasets such as CSL (`split_i` or `split_ii`) |
| `rgb_subdir` | `str` | adapter defaults | Relative RGB data directory inside a local release such as CSL |
| `corpus_file` | `str` | `""` | Sentence text file for corpora such as CSL |
| `split_spec_file` | `str` | `""` | Optional per-sample split override TSV for datasets such as CSL |
| `class_id_file` | `str` | `""` | Optional class correspondence CSV for datasets such as AUTSL |
| `train_labels_file` / `val_labels_file` / `test_labels_file` | `str` | `""` | Optional explicit split-label CSV overrides for datasets such as AUTSL |
| `subset` | `int` | `0` | Optional class-count subset for datasets such as WLASL and MS-ASL |
| `train_signers` / `val_signers` / `test_signers` | `list[int]` | adapter defaults | Explicit signer groups for signer-based datasets such as LSA64 |
| `class_map_file` | `str` | `""` | Optional class-id to gloss mapping TSV; LSA64 ships a bundled map and SLoVo can use an explicit map |
| `class_map_mode` | `str` | adapter defaults | Class ID handling for datasets such as SLoVo (`derive`, `bundled`, or `none`) |
| `allow_missing_class_map` | `bool` | `false` | Allow manifest generation without class labels when a class map is unavailable |
| `text_processing` | `dict` | adapter defaults | Text cleanup options for transcript-derived manifests |
| `sign_language` | `str` | `""` | Sign-language code written to `video2parts` metadata |
| `spoken_language` | `str` | `""` | Spoken-language code written to `video2parts` metadata |
| `country` | `str` | `""` | Country code written to `video2parts` metadata |
| `license` | `str` | `""` | Dataset license identifier written to `video2parts` metadata |

Known mismatch: `configs/jobs/how2sign/parts.yaml` supplies these four
metadata keys and `video2parts` reads them, but `How2SignSourceConfig` currently
forbids them as extras when the dataset stage begins. That shipped job cannot
complete until the metadata fields are admitted by the adapter or moved out of
`dataset.source`.

Relative file paths in `dataset.source`, such as `video_ids_file`,
`manifest_csv`, `manifest_tsv`, `bbox_json`, `annotations_csv`,
`annotations_dir`, `metadata_json`, `release_dir`,
`class_map_file`, `class_id_file`, `train_labels_file`, `val_labels_file`,
`test_labels_file`, `corpus_file`, `split_spec_file`, `metadata_file`,
`annotation_root`, and `subtitles_root`, are resolved from the project root.

## `processing`

<!-- BEGIN GENERATED: processing -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `true` | — | Run the selected processing stage. |
| `processor` | `"video2pose" \| "video2crop" \| "video2compression" \| "video2parts"` | `"video2pose"` | — | Main processor implementation. |
| `detection` | `"yolo" \| "mediapipe" \| "mmdet" \| "null"` | `"null"` | — | Person detection backend. |
| `pose` | `"mediapipe" \| "mmpose" \| null` | `null` | — | Pose backend required by video2pose and video2parts. |
| `sample_rate` | `float \| null` | `0.5` | — | Frame sampling mode: null native, fraction ratio, or absolute FPS. |
| `max_workers` | `int` | `1` | > 0 | Worker count used by post-processing and compression. |
| `detection_config` | `YOLODetectionConfig \| MMDetDetectionConfig \| MediaPipeDetectionConfig \| dict \| null` | `null` | — | Configuration for the selected detection backend. |
| `pose_config` | `MediaPipePoseConfig \| MMPosePoseConfig \| dict \| null` | `null` | — | Configuration for the selected pose backend. |
| `video_config` | `VideoProcessingConfig \| null` | `null` | — | Video crop or compression settings. |
| `parts_config` | `PartsProcessingConfig \| dict \| null` | `null` | — | Face, hand, and body-stream settings for video2parts. |
<!-- END GENERATED: processing -->

`video2pose` requires both `processing.pose` and `processing.pose_config`.
`video2parts` requires `processing.pose: mediapipe` and `processing.pose_config`.
`video2crop` auto-fills a default `processing.video_config` when omitted.

### `sample_rate` semantics

`processing.sample_rate` uses three modes:

- `null` — keep the native video FPS
- `0 < value < 1` — keep that fraction of frames uniformly
- `value >= 1` — downsample to that absolute FPS

Examples:

```yaml
processing:
  sample_rate: null   # keep native FPS
```

```yaml
processing:
  sample_rate: 0.5    # keep half the frames
```

```yaml
processing:
  sample_rate: 24.0   # downsample to 24 FPS
```

If the source video FPS is lower than an absolute target, the pipeline keeps the
source FPS. It does not upsample or invent frames.

## `processing.detection_config`

### YOLO

<!-- BEGIN GENERATED: processing.detection_config.yolo -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model` | `str` | `"yolo11m.pt"` | — | YOLO model alias or local weights path. |
| `device` | `str` | `"cpu"` | — | Ultralytics inference device, for example cpu or cuda:0. |
| `confidence_threshold` | `float` | `0.5` | — | Minimum person detection confidence. |
| `min_bbox_area` | `float` | `0.05` | — | Minimum detection box area as a fraction of frame area. |
| `batch_size` | `int` | `16` | > 0 | Frames per detector inference batch. |
| `allow_download` | `bool` | `true` | — | Allow download of missing official YOLO weights. |
| `weights_dir` | `str \| null` | `null` | — | Optional directory used to resolve local YOLO weights. |
<!-- END GENERATED: processing.detection_config.yolo -->

### MMDet

<!-- BEGIN GENERATED: processing.detection_config.mmdet -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `det_model_config` | `str` | `required` | — | MMDetection model config path. |
| `det_model_checkpoint` | `str` | `required` | — | MMDetection checkpoint path or URL. |
| `device` | `str` | `"cuda:0"` | — | MMDetection inference device. |
| `batch_size` | `int` | `16` | > 0 | Frames per detector inference batch. |
<!-- END GENERATED: processing.detection_config.mmdet -->

### MediaPipe

<!-- BEGIN GENERATED: processing.detection_config.mediapipe -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `min_detection_confidence` | `float` | `0.5` | — | Minimum MediaPipe person detection confidence. |
<!-- END GENERATED: processing.detection_config.mediapipe -->

## `processing.pose_config`

### MediaPipe

<!-- BEGIN GENERATED: processing.pose_config.mediapipe -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model_complexity` | `int` | `1` | — | MediaPipe Holistic model complexity. |
| `min_detection_confidence` | `float` | `0.5` | — | Minimum initial landmark detection confidence. |
| `min_tracking_confidence` | `float` | `0.5` | — | Minimum landmark tracking confidence. |
| `refine_face_landmarks` | `bool` | `true` | — | Enable refined face landmarks, producing 553 total points. |
| `batch_size` | `int` | `16` | — | Frames passed to the pose backend per pipeline batch. |
<!-- END GENERATED: processing.pose_config.mediapipe -->

### MMPose

<!-- BEGIN GENERATED: processing.pose_config.mmpose -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `pose_model_config` | `str` | `required` | — | MMPose model config path or package resource. |
| `pose_model_checkpoint` | `str` | `required` | — | MMPose checkpoint path or URL. |
| `device` | `str` | `"cuda:0"` | — | MMPose inference device. |
| `batch_size` | `int` | `16` | — | Frames passed to the pose backend per pipeline batch. |
<!-- END GENERATED: processing.pose_config.mmpose -->

## `processing.video_config`

<!-- BEGIN GENERATED: processing.video_config -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `codec` | `str` | `"libx264"` | — | FFmpeg encoder used for crop or compression output. |
| `padding` | `float` | `0.0` | — | Fractional padding added around a detected crop box. |
| `resize` | `list[int] \| null` | `null` | — | Optional output size as [width, height]; compression forbids resizing. |
| `crf` | `int` | `20` | ≥ 0, ≤ 63 | Constant-quality value used by video compression; crop uses 15. |
| `preset` | `str` | `"medium"` | — | Encoder speed/efficiency preset in codec-specific form. |
| `aq_strength` | `int` | `8` | ≥ 1, ≤ 15 | NVENC spatial and temporal adaptive-quantization strength. |
| `nvenc_gpu` | `int \| null` | `null` | ≥ 0 | Physical NVIDIA GPU index; null lets FFmpeg choose. |
| `max_bitrate_ratio` | `float \| null` | `0.8` | > 0.0 | Maximum output bitrate as a fraction of source bitrate. |
| `min_video_reduction_ratio` | `float` | `0.1` | ≥ 0.0, < 1.0 | Minimum size reduction required to keep a re-encode. |
| `encoding_timeout_seconds` | `int` | `3600` | > 0 | Per-video FFmpeg encoding timeout in seconds. |
<!-- END GENERATED: processing.video_config -->

`preset` is validated against the codec at config load, because ffmpeg only
rejects a mismatch once the encode starts:

| Codec | Preset form | Sent to ffmpeg as | Example |
|---|---|---|---|
| `*_nvenc` | `p1`–`p7` (`p1` fastest, `p7` best) | `-preset` | `p7` |
| `libsvtav1` | integer `-2`–`13` (**`13` fastest**, the reverse of x264) | `-preset` | `8` |
| `libaom-av1` | integer `0`–`8` (**`8` fastest**) | `-cpu-used` | `6` |
| `libvpx-vp9` | integer `-8`–`8` (**`8` fastest**) | `-cpu-used` | `4` |
| `libx264`, `libx265` | x264 names | `-preset` | `medium` |

`libaom-av1` and `libvpx-vp9` have no `-preset` option at all, so `preset` is
emitted as `-cpu-used` for them. These two also reject a bitrate ceiling unless
an explicit `-b:v` accompanies it, so `max_bitrate_ratio` is applied by setting
the bitrate target *to* the ceiling (constrained quality) rather than to `0`.

The CRF, AQ, GPU, bitrate-ratio, reduction-gate and timeout controls are read by
`video2compression`. `video2crop` keeps its own near-lossless quality target
(`-crf`/`-cq 15`) rather than reading `crf`, but it does honour `codec`,
`preset`, `padding` and `resize`.

The schema defaults above remain generic for software-encoder and crop-job
compatibility. The bundled
[`configs/jobs/tools/compression.yaml`](../../configs/jobs/tools/compression.yaml)
job overrides them with the evaluated default: `av1_nvenc`, CQ34, p7, AQ8,
CUDA 0, and a 0.52× source-bitrate ceiling.

## `processing.parts_config`

<!-- BEGIN GENERATED: processing.parts_config -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `crop_size` | `int` | `224` | > 0 | Square size in pixels for face and hand streams. |
| `bbox_scale` | `float` | `1.2` | > 0 | Scale applied to face and hand landmark boxes. |
| `codec` | `str` | `"mp4v"` | — | OpenCV FourCC used for part-stream MP4 files. |
| `missing_value` | `float` | `-1.0` | — | Sentinel for missing body pose values and boxes. |
<!-- END GENERATED: processing.parts_config -->

## `post_processing`

<!-- BEGIN GENERATED: post_processing -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | — | Run landmark post-processing. |
| `normalize` | `NormalizeConfig \| null` | `null` | — | Landmark normalization settings. |
<!-- END GENERATED: post_processing -->

## `post_processing.normalize`

<!-- BEGIN GENERATED: post_processing.normalize -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `mode` | `"isotropic_3d" \| "xy_isotropic_z_minmax"` | `"xy_isotropic_z_minmax"` | — | Coordinate normalization algorithm. |
| `remove_z` | `bool` | `false` | — | Drop the z coordinate after normalization. |
| `select_keypoints` | `bool` | `true` | — | Apply a preset or explicit keypoint selection before flattening. |
| `keypoint_preset` | `str \| null` | `null` | — | Named built-in keypoint selection preset. |
| `keypoint_indices` | `list[int] \| null` | `null` | — | Explicit keypoint indices used when no preset is selected. |
| `mask_empty_frames` | `bool` | `true` | — | Replace all-zero frames with the missing-value sentinel. |
| `mask_low_confidence` | `bool` | `false` | — | Mask landmarks below the visibility threshold. |
| `visibility_threshold` | `float` | `0.3` | — | Visibility cutoff used by low-confidence masking. |
| `missing_value` | `float` | `-999.0` | — | Sentinel written for masked coordinates. |
<!-- END GENERATED: post_processing.normalize -->

## `output`

<!-- BEGIN GENERATED: output -->
| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `true` | — | Run WebDataset packaging. |
| `max_shard_count` | `int` | `10000` | > 0 | Maximum samples per tar shard. |
| `max_shard_size` | `int \| null` | `null` | > 0 | Approximate maximum payload bytes per shard. |
<!-- END GENERATED: output -->

## CLI Examples

```bash
# Change number of normalization workers
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8

# Change normalization behavior
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override post_processing.normalize.remove_z=true

# Override run name for isolated outputs
python -m signdata run configs/jobs/youtube_asl/video.yaml \
  --run-name privacy_v1

# List built-in keypoint presets
python -m signdata run --list-presets
```

## See Also

- [Architecture](../concepts/architecture.md) -- runner, registries, and package layout
- [Pipeline Stages](stages.md) -- stage behavior and outputs
- [Installation Guide](../guides/installation.md) -- environment setup and MMPose dependencies
