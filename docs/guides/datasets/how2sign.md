# How2Sign

80+ hours of instructional "how-to" videos with continuous ASL, recorded in a controlled environment with professional signers ([Duarte et al., CVPR 2021](https://how2sign.github.io/)).

**Default pose job:** `dataset.download (validation only) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/how2sign/mediapipe.yaml
```

**Default video job:** `dataset.download (validation only) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/how2sign/video.yaml
```

**Setup:**
1. Download the dataset from [how2sign.github.io](https://how2sign.github.io/)
2. Place videos in the `videos` path (default: `dataset/how2sign/videos/`)
3. Place the alignment CSV (e.g. `how2sign_realigned_val.csv`) at `paths.manifest` or `dataset.source.manifest_csv`

The How2Sign dataset adapter uses `dataset.download` as a validation step for
local files; it does not fetch remote data.

