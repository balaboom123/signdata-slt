# MS-ASL

Large-scale isolated ASL dataset with signer-diverse lexical clips ([Joze and Koller, CVPR 2019](https://arxiv.org/abs/1812.01053)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/msasl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/msasl/video.yaml
```

**Setup:**
1. Download the official MS-ASL annotation release from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=100121)
2. Place `MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`, and `MSASL_classes.json` under `dataset/msasl/annotations/` or override `dataset.source.annotations_dir`
3. Place local clips under `paths.videos` (default: `dataset/msasl/videos/`); both flat layouts and nested-by-class layouts are supported
4. The provided base config keeps `dataset.source.download_mode: validate`; optionally tune `dataset.source.split`, `dataset.source.subset`, and `dataset.source.availability_policy`

The MS-ASL dataset adapter supports two acquisition modes:

- `download_mode: validate` treats files under `paths.videos` as the local clip corpus and writes per-sample `REL_PATH` values into the manifest.
- `download_mode: download_missing` extracts YouTube IDs from the selected split JSON files and downloads any missing videos into `paths.videos`.

