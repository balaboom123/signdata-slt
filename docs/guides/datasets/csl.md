# CSL

Continuous Chinese Sign Language dataset released by USTC with 100 sentence prompts and 50 signers. The release page describes RGB/depth/skeleton modalities, while the benchmark papers commonly use one RGB clip per signer-sentence pair for Split I / Split II evaluation ([USTC CSL release](https://ustc-slr.github.io/datasets/2015_csl/)).

**Default pose job:** `dataset.download (validate/materialize local release) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/csl/mediapipe.yaml
```

**Default video job:** `dataset.download (validate/materialize local release) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/csl/video.yaml
```

**Setup:**
1. Request access from the official USTC page: download the release agreement, sign it, and email it as instructed at [ustc-slr.github.io/datasets/2015_csl](https://ustc-slr.github.io/datasets/2015_csl/).
2. Unpack the release under `dataset/csl/` so the corpus file lives at `dataset/csl/corpus.txt` and RGB data lives under `dataset/csl/color/`, or override `dataset.source.release_dir`.
3. Run the normal CSL job. `dataset.download` is intentionally simple:
   it only validates a local release and, when needed, materializes frame folders into `paths.videos`.
4. `dataset.manifest` follows the paper-aligned split logic:
   `split_i` uses signers 1-40 for train and 41-50 for test; `split_ii` uses sentences 1-94 for train and 95-100 for test.

The shipped CSL configs default to `dataset.source.prepare_mode: materialize_missing`, so both common local layouts are supported:

- RGB video clips already present under `dataset.source.rgb_subdir` such as `dataset/csl/color/000000/*.mp4`
- per-sample frame folders such as `dataset/csl/color/000000/<sample>/000001.jpg`, which are converted to `.mp4` clips before preprocessing

The adapter targets the **continuous 2015 CSL release** and currently uses only the RGB modality in this preprocessing pipeline. The native depth and Kinect skeleton files from the release are not ingested directly by SignDATA.

