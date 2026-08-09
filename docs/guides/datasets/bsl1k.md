# BSL-1K

Compatibility lexical view built from the public BOBSL release rather than a separate downloader ([BSL-1K project page](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/bsl1k/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/bsl1k/video.yaml
```

The shipped BSL-1K configs reuse `dataset.source.release_dir: dataset/bobsl` and force `dataset.source.view: isolated_signs`, so outputs are organized under `dataset/bsl1k/` while the raw release and default video root stay under `dataset/bobsl/`.

