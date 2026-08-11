# WLASL

Word-level ASL dataset with 2,000 glosses and 12,000+ isolated sign videos ([Li et al., WACV 2020](https://dxli94.github.io/WLASL/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/wlasl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/wlasl/video.yaml
```

**Setup:**
1. Download and preprocess WLASL clips with the official [start-kit repository](https://github.com/dxli94/WLASL)
2. Place `WLASL_v0.3.json` at `dataset/wlasl/WLASL_v0.3.json` or override `dataset.source.metadata_json`
3. Place one preprocessed clip per `video_id` under `paths.videos` (default: `dataset/wlasl/videos/`)
4. The provided base config keeps `dataset.source.download_mode: validate` for local preprocessed clips; optionally tune `dataset.source.split`, `dataset.source.subset`, and `dataset.source.availability_policy`

The WLASL dataset adapter supports two acquisition modes:

- `download_mode: validate` treats files under `paths.videos` as preprocessed clips. Manifest rows keep the original `FRAME_START` / `FRAME_END` metadata, but `START=0.0` and `END` is taken from the isolated clip duration when available.
- `download_mode: download_missing` fetches missing raw source videos from each instance `url` in `WLASL_v0.3.json` and keeps source-aligned `START` / `END` timing in the manifest.

Acquisition maps the obsolete `aslsignbank.haskins.yale.edu` hostname to its
certificate-valid canonical host, `aslsignbank.com`; TLS verification remains
enabled.

