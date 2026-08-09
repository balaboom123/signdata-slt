# Output Formats

All processor artifacts are scoped by `run_name` under
`{paths.output}/{run_name}`. WebDataset shards are scoped independently under
`{paths.webdataset}/{run_name}`.

## `video2pose`

Raw landmark clips are written to:

```text
{paths.output}/{run_name}/raw/{SAMPLE_ID}.npy
```

The saved array is `float32` with shape `(T, K, 4)`:

| Axis/channel | Meaning |
|---|---|
| `T` | Frames selected from `[START, END)` by `processing.sample_rate` |
| `K` | Landmark count: 553 refined MediaPipe, 543 unrefined MediaPipe, or 133 MMPose COCO WholeBody |
| channel `0` | `x`; normalized by frame width for both backends |
| channel `1` | `y`; normalized by frame height for both backends |
| channel `2` | `z`; MediaPipe depth, MMPose model depth, or zero for a 2D MMPose model |
| channel `3` | Visibility/confidence; face and hand MediaPipe points default to `1.0` when the API exposes no visibility |

MediaPipe keypoint order is 33 pose, 468 or 478 face, 21 left hand, then 21
right hand. A failed frame is represented by a `(K, 4)` zero array. A sample is
not saved at all when every value in the complete clip is zero.

### Normalized pose output

`post_processing.normalize` reads raw arrays and writes:

```text
{paths.output}/{run_name}/normalized/{SAMPLE_ID}.npy
```

Normalized files are flattened `float32` arrays with shape `(T, K' * C)`, not
`(T, K', C)`:

- `K'` is `K` unless keypoint selection resolves a preset or explicit index
  list. The shipped MediaPipe and MMPose presets select 85 points; the
  unrefined MediaPipe preset selects 83.
- `C` is 3 (`x`, `y`, `z`) by default and 2 when `remove_z: true`.
- Visibility is used for optional masking and is then removed.
- Missing coordinates retain the configured `missing_value` sentinel.

For the shipped YouTube-ASL MediaPipe job, the normalized shape is `(T, 255)`:
85 selected keypoints × 3 coordinates. With `remove_z: true`, it is `(T, 170)`.
With `select_keypoints: false`, refined MediaPipe output is `(T, 1659)` or
`(T, 1106)` depending on whether `z` is retained.

## `video2crop`

Cropped clips are written to:

```text
{paths.output}/{run_name}/raw/{SAMPLE_ID}.mp4
```

The processor decodes the segment once for detection, computes one union box,
and then invokes FFmpeg for the crop. It applies `video_config.padding`, optional
`video_config.resize`, and the effective sample FPS.

Crop encoding has a fixed near-lossless target independent of
`video_config.crf`: software encoders receive `-crf 15`; NVENC encoders receive
`-rc vbr -cq 15 -b:v 0`. The configured codec and preset still apply. Audio is
removed with `-an`.

## `video2compression`

This processor works once per unique source video rather than once per manifest
row. It writes a complete mirror under:

```text
{paths.output}/{run_name}/compressed/{REL_PATH without its old suffix}.mp4
```

When `REL_PATH` is absent, the filename is derived from legacy `VIDEO_NAME` or
`VIDEO_ID`. Directory components in `REL_PATH` are preserved.

- Successful re-encodes are always `.mp4` and preserve geometry, duration, and
  source frame cadence.
- Sources already in the target codec family, encodes that miss the configured
  size-reduction gate, and read/encode failures are passed through unchanged.
- Passthrough files are hardlinks when possible and copies otherwise. They keep
  the source extension, so a mirror can contain both `.mp4` and `.webm` files.
- The processor never deletes source files.
- `output.webdataset` does not support `video2compression`; repoint
  `paths.videos` at the verified mirror before a later processing job.

Each successful re-encode gets a sidecar named
`<output>.compression.json`. Passthrough files do not. The JSON object has
`"format": "signdata.compression.v2"` and these fields:

| Field | Meaning |
|---|---|
| `source_path` | Source path used for the encode |
| `source_codec`, `output_codec` | FFprobe codec names |
| `source_pix_fmt`, `output_pix_fmt` | FFprobe pixel formats |
| `source_bytes`, `output_bytes` | File sizes |
| `source_bitrate_bps`, `output_bitrate_bps` | Probed bitrates |
| `width`, `height`, `duration` | Output geometry and seconds; the processor verifies these against the source before promotion |
| `encoder` | Configured FFmpeg encoder name |
| `crf`, `preset`, `aq_strength`, `nvenc_gpu` | Encoder settings recorded from `processing.video_config` |

`max_bitrate_ratio`, `min_video_reduction_ratio`, and
`encoding_timeout_seconds` affect the decision but are not recorded in the
sidecar.

## `video2parts`

Each sample is a directory:

```text
{paths.output}/{run_name}/raw/{SAMPLE_ID}/
├── face.mp4
├── left_hand.mp4
├── right_hand.mp4
├── pose.npz
└── meta.json
```

The three synchronized videos contain square RGB crops at
`parts_config.crop_size` (224 × 224 by default) and the effective sampled FPS.
A missing crop is a black frame; hand boxes may reuse the previous valid box.

### `pose.npz`

Let `T` be the number of sampled frames. The compressed NPZ contains:

| Key | Dtype | Shape | Meaning |
|---|---|---:|---|
| `body_pose` | `float32` | `(T, 14)` | Seven upper-body points flattened as `(x, y)` after SignMusketeers-style normalization |
| `frame_time_s` | `float32` | `(T,)` | Absolute source timestamps in seconds |
| `frame_index` | `int64` | `(T,)` | Rounded source-frame indices derived from timestamp × source FPS |
| `face_bbox_xyxy` | `float32` | `(T, 4)` | Face box in source-frame pixel coordinates |
| `left_bbox_xyxy` | `float32` | `(T, 4)` | Left-hand box in source-frame pixel coordinates |
| `right_bbox_xyxy` | `float32` | `(T, 4)` | Right-hand box in source-frame pixel coordinates |
| `valid` | `bool` | `(T, 4)` | Validity flags ordered face, left hand, right hand, body pose |

Invalid body pose uses the previous valid vector when available, otherwise
fourteen `parts_config.missing_value` entries. Missing boxes use the same
sentinel four times.

### `meta.json`

The metadata format is `signdata.parts.v1`. It records `sample_id`, `video_id`,
`source_dataset`, `sign_language`, `spoken_language`, `country`, `start`, `end`,
`source_fps`, `sample_rate`, `frame_count`, `crop_size`, `bbox_scale`,
`pose_backend`, `pose_landmark_layout`, `body_pose_layout`, `license`, and
`source_url`. In this file, `sample_rate` is the effective output FPS, not
necessarily the literal ratio from the job config.

The current writer reads `source_url` from a manifest column named `URL`, while
built-in WLASL and MS-ASL manifests emit `SOURCE_URL`; their value is therefore
empty in `meta.json` unless an input manifest also supplies `URL`.

## `output.webdataset`

Shards are uncompressed tar files named:

```text
{paths.webdataset}/{run_name}/shard-000000.tar
{paths.webdataset}/{run_name}/shard-000001.tar
...
```

A new shard starts before the next sample when `max_shard_count` is reached or
when the current payload size has reached `max_shard_size`. Tar members have a
fixed modification time of zero.

| Processor | Members for each `{SAMPLE_ID}` |
|---|---|
| `video2pose` | `.npy`, `.txt`, `.json`; normalized NPY is preferred over raw NPY |
| `video2crop` | `.mp4`, `.txt`, `.json` |
| `video2parts` | `.face_mp4`, `.left_hand_mp4`, `.right_hand_mp4`, `.pose_npz`, `.json`, `.txt` |
| `video2compression` | Unsupported |

For pose and crop samples, JSON contains `video_id`, `sample_id`, `start`,
`end`, and `processor`. Parts samples package the existing `meta.json` instead.
`.txt` contains `TEXT`, or an empty byte string when that column/value is
missing. A sample is skipped unless every required processor artifact exists.

### Minimal reader

The `webdataset` reader is optional and is not installed by SignDATA:

```bash
python -m pip install webdataset
```

```python
import io
import json

import numpy as np
import webdataset as wds

samples = wds.WebDataset("dataset/youtube_asl/webdataset/default/shard-000000.tar")
for sample in samples:
    sample_id = sample["__key__"]
    landmarks = np.load(io.BytesIO(sample["npy"]))
    text = sample["txt"].decode("utf-8")
    metadata = json.loads(sample["json"])
    print(sample_id, landmarks.shape, text, metadata["processor"])
    break
```

## See Also

- [Manifest reference](manifest.md) -- input table contract
- [Pipeline stages](stages.md) -- when each artifact is produced
- [Configuration reference](configuration.md) -- processor options
