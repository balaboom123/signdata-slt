# Pipeline Stages

The pipeline has 6 stages. Which ones run is controlled by `pipeline.steps` in your config.

Pose mode typically runs: `download → manifest → extract → normalize → webdataset`
Video mode typically runs: `download → manifest → clip_video → webdataset`

## Stage Summary

| Stage | Mode | Input | Output | Skips existing? |
|---|---|---|---|---|
| `download` | both | video ID list | `.mp4` + `.json` transcript per video | Yes — by file presence |
| `manifest` | both | transcript `.json` files | manifest CSV | No — always rebuilds |
| `extract` | pose | manifest + `.mp4` videos | `(T, K, 4)` `.npy` per segment | Yes — always skips existing `.npy` |
| `normalize` | pose | `(T, K, 4)` `.npy` files | `(T, K'×C)` flattened `.npy` | Yes — `processing.skip_existing` |
| `clip_video` | video | manifest + `.mp4` videos | `.mp4` clip per segment | Yes — always skips existing `.mp4` |
| `webdataset` | both | `.npy` or `.mp4` files + manifest | `.tar` shards | No — overwrites from shard-000000 but does not remove old shards |

---

## 1. download

Downloads YouTube videos and transcripts using yt-dlp and `youtube-transcript-api`.

- **Input:** `download.video_ids_file` (one video ID per line)
- **Output:** `{paths.videos}/{video_id}.mp4`, `{paths.transcripts}/{video_id}.json`
- **Key config:** `download.format`, `download.rate_limit`, `download.languages`, `download.concurrent_fragments`
- **Skip existing:** Skips a video if `{paths.videos}/{video_id}.mp4` already exists on disk.
- **Notes:** Only used by YouTube-ASL; How2Sign requires pre-downloaded data.
- **→** [download config reference](configuration.md#download)

---

## 2. manifest

Parses transcript JSON files into a TSV manifest with one row per segment.

- **Input:** `{paths.transcripts}/*.json`
- **Output:** `{paths.manifest}` (CSV with columns `VIDEO_NAME`, `SENTENCE_NAME`, `START_REALIGNED`, `END_REALIGNED`, `SENTENCE`)
- **Key config:** `manifest.max_text_length`, `manifest.min_duration`, `manifest.max_duration`
- **Skip existing:** Always rebuilds the manifest from all transcript files.
- **Notes:** Applies Unicode normalization (ftfy), text cleaning, and duration/length filtering. Segment IDs follow the pattern `{video_id}-{index}`.
- **→** [manifest config reference](configuration.md#manifest)

---

## 3. extract

Extracts per-frame pose landmarks from video segments defined in the manifest.

- **Input:** `{paths.videos}/{video_name}.mp4` + manifest
- **Output:** `{paths.landmarks}/{sentence_name}.npy` with shape `(T, K, 4)` where channels are `[x, y, z, visibility]`
- **Key config:** `extractor.name`, `extractor.max_workers`, `processing.target_fps`, `processing.frame_skip`, `processing.accept_fps_range`
- **Keypoint counts (K):**
  - MediaPipe (refined face): **553** = 33 pose + 478 face + 21 left hand + 21 right hand
  - MediaPipe (unrefined): **543** = 33 pose + 468 face + 21 left hand + 21 right hand
  - MMPose RTMPose3D: **133** (COCO WholeBody)
- **Skip existing:** Always skips segments where `{paths.landmarks}/{sentence_name}.npy` already exists, regardless of `processing.skip_existing`. To re-extract, delete the existing `.npy` files.
- **Notes:** Always outputs all keypoints; reduction happens in the normalize step. Uses `ProcessPoolExecutor` for parallel extraction. MMPose requires CUDA.
- **→** [extractor config reference](configuration.md#extractor) · [processing config reference](configuration.md#processing)

---

## 4. normalize

Applies keypoint reduction, visibility masking, coordinate normalization, and flattening.

- **Input:** `{paths.landmarks}/*.npy` with shape `(T, K, 4)`
- **Output:** `{paths.normalized}/*.npy` with shape `(T, K'*C)` flattened, where `K'` is the reduced keypoint count and `C` is 3 (visibility is always stripped); `C` = 2 when `remove_z=true`
- **Key config:** `normalize.mode`, `normalize.reduction`, `normalize.keypoint_indices`, `normalize.remove_z`, `normalize.visibility_threshold`, `normalize.unvisible_value`
- **Processing pipeline:**
  1. **Reduction** -- select a subset of keypoints (default: 85 for MediaPipe refined/MMPose, 83 for MediaPipe unrefined). Custom indices can be specified via `keypoint_indices`.
  2. **Visibility masking** -- frame-level: replace all-zero frames with `unvisible_value`; landmark-level: replace landmarks below `visibility_threshold`.
  3. **Normalization** -- `xy_isotropic_z_minmax` normalizes XY isotropically and Z via min-max; `isotropic_3d` normalizes XYZ with a single scale factor.
  4. **Z removal** -- optionally drop the z-coordinate.
  5. **Flatten** -- reshape from `(T, K', C)` to `(T, K'*C)`.

- **Skip existing:** When `processing.skip_existing=true`, skips segments where `{paths.normalized}/{sentence_name}.npy` already exists.
- **→** [normalize config reference](configuration.md#normalize)

---

## 5. clip_video

Clips full-length videos into segments using ffmpeg, based on the manifest timestamps.

- **Input:** `{paths.videos}/{video_name}.mp4` + manifest
- **Output:** `{paths.clips}/{sentence_name}.mp4`
- **Key config:** `clip_video.codec`, `clip_video.resize`
- **Skip existing:** Always skips segments where `{paths.clips}/{sentence_name}.mp4` already exists, regardless of `processing.skip_existing`. To re-clip, delete the existing `.mp4` files.
- **Notes:** Uses `codec: copy` by default for fast stream-copy (no re-encoding). Set `codec: libx264` if you need re-encoding. Optional `resize: [width, height]` for rescaling.
- **→** [clip_video config reference](configuration.md#clip_video)

---

## 6. webdataset

Packages processed outputs into tar shards for efficient data loading with the `webdataset` library.

- **Input:** Normalized `.npy` files (pose mode) or clipped `.mp4` files (video mode) + manifest
- **Output:** `{paths.webdataset}/shard-{NNNNNN}.tar`
- **Key config:** `webdataset.max_shard_count`, `webdataset.max_shard_size`
- **Skip existing:** Always overwrites shards starting from `shard-000000.tar`. **Caution:** old higher-numbered shards from a previous run are not deleted. Clear `{paths.webdataset}` manually before re-running if the new run produces fewer shards than the previous one.
- **Shard contents per sample:**
  - `__key__` -- sentence name
  - `.npy` or `.mp4` -- landmark array or video clip
  - `.txt` -- transcript text
  - `.json` -- metadata (video ID, sentence name, start/end times, extractor, mode)
- **Output example:**
  ```
  dataset/youtube_asl/webdataset/
  ├── shard-000000.tar        # up to 10,000 samples per shard
  ├── shard-000001.tar
  └── ...
  ```
  Each `.npy` sample has shape `(T, K'×C)` where `T` = frames, `K'` = reduced keypoints, `C` = 3 (or 2 with `remove_z`).
- **→** [webdataset config reference](configuration.md#webdataset)

---

## See Also

- [Architecture](architecture.md) -- pipeline flow and registry system
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Datasets](datasets.md) -- dataset-specific setup guides
