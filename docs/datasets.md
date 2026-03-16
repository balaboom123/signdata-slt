# Datasets

## YouTube-ASL

A large-scale, open-domain ASL-English parallel corpus with 11,000+ YouTube videos and 73,000+ segments ([Uthus et al., 2023](https://arxiv.org/abs/2306.15162)).

**Full pipeline:** `download → manifest → extract → normalize → webdataset`

```bash
python -m sign_prep configs/youtube_asl/pose_mediapipe.yaml
```

Requires `download.video_ids_file` pointing to the video ID list (included at `assets/youtube-asl_youtube_asl_video_ids.txt`). The download step fetches videos via yt-dlp and transcripts via `youtube-transcript-api`.

## How2Sign

80+ hours of instructional "how-to" videos with continuous ASL, recorded in a controlled environment with professional signers ([Duarte et al., CVPR 2021](https://how2sign.github.io/)).

**Pipeline:** `extract → normalize → webdataset` for the standard pose configs

```bash
python -m sign_prep configs/how2sign/pose_mediapipe.yaml
```

**Setup:**
1. Download the dataset from [how2sign.github.io](https://how2sign.github.io/)
2. Place videos in the `videos` path (default: `dataset/how2sign/videos/`)
3. Place the alignment CSV (e.g. `how2sign_realigned_val.csv`) at the `manifest` path

The How2Sign dataset class rejects `download` steps. Experiment configs may still
include `manifest` when they reuse an existing alignment CSV and do not rely on
the YouTube transcript-building flow.

## Adding a New Dataset

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-dataset) for step-by-step instructions and code examples.

---

## See Also

- [Pipeline Stages](pipeline-stages.md) -- what each stage does and its I/O
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Research-Aligned Preprocessing](research-preprocessing.md) -- paper-aligned methodology notes
