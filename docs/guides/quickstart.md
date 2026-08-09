# Quickstart

This offline smoke run exercises a real How2Sign job path from local source
validation through manifest loading, video processing, and WebDataset output. It
uses one generated two-second clip, so it does not require dataset credentials,
a GPU, or a model download. FFmpeg is required.

## Install

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv venv
source venv/bin/activate
python -m pip install -e .
ffmpeg -version
```

## Create one local sample

```bash
mkdir -p dataset/quickstart/videos dataset/quickstart/text
ffmpeg -y -f lavfi -i testsrc2=size=320x240:rate=10 -t 2 \
  -c:v libx264 -pix_fmt yuv420p dataset/quickstart/videos/smoke.mp4
printf 'SAMPLE_ID\tVIDEO_ID\tSTART\tEND\tTEXT\nsmoke-000\tsmoke\t0\t2\thello\n' \
  > dataset/quickstart/text/manifest.tsv
```

The manifest is tab-separated. `START` and `END` are seconds.

## Run to shards

```bash
python -m signdata run configs/jobs/how2sign/video.yaml \
  --run-name smoke \
  --override \
    paths.root=dataset/quickstart \
    paths.videos=dataset/quickstart/videos \
    paths.manifest=dataset/quickstart/text/manifest.tsv \
    dataset.source.manifest_csv=dataset/quickstart/text/manifest.tsv \
    'processing.detection="null"' \
    processing.detection_config=null \
    processing.sample_rate=null \
    processing.max_workers=1
```

The null detector deliberately treats the complete frame as the signer box. It
keeps this smoke test deterministic; production How2Sign video jobs use the
configured person detector.

A successful run reports one processed sample and one shard. It creates:

```text
dataset/quickstart/
├── videos/smoke.mp4
├── text/manifest.tsv
├── output/smoke/raw/smoke-000.mp4
└── webdataset/smoke/shard-000000.tar
```

Verify the shard:

```bash
tar -tf dataset/quickstart/webdataset/smoke/shard-000000.tar
```

It contains exactly three members:

```text
smoke-000.mp4
smoke-000.txt
smoke-000.json
```

Replace the generated clip and manifest with the real How2Sign release when
moving beyond the smoke run; see the [How2Sign guide](datasets/how2sign.md).
For pose arrays instead of cropped video, use the MediaPipe job after supplying
real signer video.

## Next

- [Dataset guides](datasets/index.md) -- configure a supported corpus
- [Output formats](../reference/outputs.md) -- inspect arrays and shard keys
- [Troubleshooting](troubleshooting.md) -- common environment failures
