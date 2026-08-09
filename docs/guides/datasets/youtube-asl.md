# YouTube-ASL

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

