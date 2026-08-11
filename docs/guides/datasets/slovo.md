# SLoVo

Russian Sign Language isolated-sign dataset with trimmed RGB videos and an
`annotations.csv` file ([SLoVo GitHub](https://github.com/hukenovs/slovo)).

**First-stage config:**

```bash
python -m signdata run configs/base/datasets/slovo.yaml \
  --override output.enabled=false
```

No complete SLoVo processing job is shipped. This is intentionally a
manifest-only command: omitted paths derive from `dataset/slovo`,
`processing.enabled` defaults to `false`, and the override disables shard
output. Create a job under `configs/jobs/` to process the resulting manifest.

**Setup:**
1. Download and extract the official trimmed SLoVo release under `dataset/slovo/`, or override `dataset.source.release_dir`.
2. Keep the official `annotations.csv` in the release root, or override `dataset.source.annotations_csv`.
3. Keep videos named by `attachment_id`. The official archive stores them under `dataset/slovo/train/<attachment_id>.mp4` and `dataset/slovo/test/<attachment_id>.mp4`; a repackaged flat release root is also accepted.

The SLoVo adapter auto-detects the official tab-separated annotation format, validates the local release, and builds one row per annotation.
It maps the official `train` boolean to `SPLIT=train|test`, uses `text` as both
`GLOSS` and `TEXT`, and derives `CLASS_ID` values from the annotation labels by
default.

