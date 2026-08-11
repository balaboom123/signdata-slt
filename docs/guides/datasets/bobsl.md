# BOBSL

Broadcast BSL corpus released with subtitle-aligned interpreter videos and manual isolated-sign annotations ([Albanie et al., ICCV 2021](https://arxiv.org/abs/2111.03635)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/bobsl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/bobsl/video.yaml
```

**Setup:**
1. Apply for access through the [official BOBSL page](https://www.robots.ox.ac.uk/~vgg/data/bobsl/), complete the BBC Terms of Use agreement, and wait for the personal download password. The interpreter-cropped MP4 release is 331 GB, so confirm archive and extraction capacity before downloading it under `dataset/bobsl/` or another `dataset.source.release_dir`.
2. Place interpreter-cropped MP4s under `paths.videos` (the shipped jobs use `dataset/bobsl/videos/`).
3. Keep the release metadata JSON (for example `metadata/subset2episode.json`) under the release root, or override `dataset.source.metadata_file`.
4. Keep subtitle files under the release root. The adapter auto-discovers the common `subtitles/manually-aligned/` and `subtitles/audio-aligned-heuristic-correction/` layouts, or you can override `dataset.source.subtitles_root`.
5. Keep manual isolated-sign annotations under the release root. The adapter auto-discovers common `annotations/` layouts, or you can override `dataset.source.annotation_root`.

The BOBSL adapter exposes two manifest views:

- `dataset.source.view=subtitle_slt` builds one row per subtitle segment with `TEXT`, `START`, and `END`.
- `dataset.source.view=isolated_signs` builds one row per manual isolated-sign annotation with `GLOSS`, `TEXT`, `CLASS_ID`, `START`, and `END`.

Available BOBSL-specific overrides:

- `dataset.source.subtitle_alignment=manual|original` chooses between manually aligned subtitles and the original/audio-aligned subtitle release.
- `dataset.source.split=train|val|test|all` filters the selected release partition.
- `dataset.source.metadata_file`, `subtitles_root`, and `annotation_root` let you point at repackaged local layouts when auto-discovery is not enough.

