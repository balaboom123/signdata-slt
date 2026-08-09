# LSA64

Argentinian Sign Language isolated-sign dataset with 64 glosses, 10 signers, and 3,200 RGB clips ([Ronchetti et al., CACIC 2016](https://facundoq.github.io/datasets/lsa64/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/lsa64/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/lsa64/video.yaml
```

**Setup:**
1. Download the official dataset from [facundoq.github.io/datasets/lsa64](https://facundoq.github.io/datasets/lsa64/)
2. Extract either the **cut** or **raw** RGB release under `dataset/lsa64/`
3. The shipped configs default to `dataset.source.variant: cut`; override with `dataset.source.variant=raw` to target the raw release
4. Keep `dataset.source.release_dir` pointed at the release root (`dataset/lsa64/`) unless you want to bypass variant resolution and point directly at a flat clip directory

The LSA64 adapter supports both official RGB layouts:

- release root with variant subdirectories such as `dataset/lsa64/cut/*.mp4` and `dataset/lsa64/raw/*.mp4`
- flat clip directories containing `*.mp4` files directly

The canonical file naming convention is `{CLASS_ID}_{SIGNER_ID}_{REPETITION_ID}.mp4`, for example `01_09_05.mp4`.

The shipped configs use `split_strategy: none` and emit `SPLIT=all` by default. For signer-independent evaluation, enable:

```bash
python -m signdata run configs/jobs/lsa64/mediapipe.yaml \
  --override dataset.source.split_strategy=community_signer_8_1_1 dataset.source.split=train
```

By default, the adapter loads the bundled `assets/lsa64_class_map.tsv` gloss map derived from the official dataset website. You can override it with `dataset.source.class_map_file`.

LSA64 is licensed under [CC BY-NC-SA 4.0](https://facundoq.github.io/datasets/lsa64/). If you use or redistribute derivatives of the dataset, the dataset authors request that you cite the official website or paper.

