# Datasets

## YouTube-ASL

A large-scale, open-domain ASL-English parallel corpus with 11,000+ YouTube videos and 73,000+ segments ([Uthus et al., 2023](https://arxiv.org/abs/2306.15162)).

**Default pose job:** `dataset.download → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml
```

**Default video job:** `dataset.download → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/youtube_asl/video.yaml
```

Requires `dataset.source.video_ids_file` pointing to the video ID list
(included at `assets/youtube-asl_youtube_asl_video_ids.txt`). The dataset
download stage fetches videos via yt-dlp and transcripts via
`youtube-transcript-api`. If transcript requests start failing with
`RequestBlocked` or `IpBlocked`, configure
`dataset.source.transcript_proxy_http` / `dataset.source.transcript_proxy_https`
or retry from a non-blocked residential IP.

## How2Sign

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

## Adding a New Dataset

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-dataset) for step-by-step instructions and code examples.

---

## See Also

- [Pipeline Stages](pipeline-stages.md) -- what each stage does and its I/O
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Research-Aligned Preprocessing](research-preprocessing.md) -- paper-aligned methodology notes
