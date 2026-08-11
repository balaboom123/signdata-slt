# OpenASL

Open-domain ASL-English translation data from online video with an official TSV
manifest and optional bounding boxes ([OpenASL GitHub](https://github.com/chevalierNoir/OpenASL)).

**First-stage config:**

```bash
python -m signdata run configs/base/datasets/openasl.yaml \
  --override output.enabled=false
```

No complete OpenASL processing job is shipped. This is intentionally a
manifest-only command: omitted paths derive from `dataset/openasl`,
`processing.enabled` defaults to `false`, and the override disables shard
output. Create a job under `configs/jobs/` to process the resulting manifest.

**Setup:**
1. Download `data/openasl-v1.0.tsv` and `data/bbox-v1.0.json` from the official OpenASL repository.
2. Place them under `dataset/openasl/data/`, or override `dataset.source.manifest_tsv` and `dataset.source.bbox_json`.
3. Keep `paths.videos` pointed at the raw-video directory. `dataset.download` fetches raw YouTube videos by `yid`; `dataset.manifest` maps `vid`, `yid`, `start`, `end`, and the configured text column into the canonical manifest.

OpenASL's official preprocessing also trims and spatially crops clips with the
provided bbox file. In this repo that happens after the first stage through the
normal video processing stages; the manifest stores timing and bbox metadata.

For long downloads, `sleep_interval_requests` paces extractor requests and
`sleep_interval` / `max_sleep_interval` add a randomized delay before each
media download. Completed source videos are skipped when acquisition resumes.
The default `stop_on_download_block: true` exits promptly on HTTP 429 or bot
verification instead of exhausting the remaining source IDs.

