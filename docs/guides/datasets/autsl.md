# AUTSL

Automatic Turkish Sign Language dataset used in the CVPR/ICCV 2021 isolated sign challenge. The release contains signer-independent `train`, `val`/`validation`, and `test` splits with paired RGB/depth clips and class-ID correspondence files.

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/autsl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/autsl/video.yaml
```

**Setup:**
1. Register or sign in on the [official AUTSL ChaLearn page](https://chalearnlap.cvc.uab.es/dataset/40/description/), accept the challenge terms, and download the encrypted train, validation, and test archives from the account-only file listings. Extract them under `dataset/autsl/`, or override `dataset.source.release_dir`. The archive keys are published on the corresponding ChaLearn data-description pages.
2. Keep the split directories under the release root, for example `dataset/autsl/train/`, `dataset/autsl/validation/`, and `dataset/autsl/test/`.
3. Ensure the release contains the class-ID correspondence CSV (for example `SignList*.csv`) plus `train_labels.csv` and `val_labels.csv`. The shipped configs assume the common challenge layout and auto-discover these files.
4. The shipped AUTSL jobs intentionally set `paths.videos: dataset/autsl` instead of `dataset/autsl/videos` because manifest rows use split-relative paths such as `train/signer0_sample1_color.mp4`.

The shipped AUTSL configs default to RGB clips via `dataset.source.modality: rgb` and keep `allow_unlabeled: false`, so unlabeled public `test` rows are skipped by default even when `dataset.source.split: all`.

Available AUTSL-specific overrides:

- `dataset.source.modality=rgb|depth` chooses the `_color.mp4` or `_depth.mp4` files.
- `dataset.source.split=train|val|test|all` filters which challenge split(s) feed the manifest.
- `dataset.source.allow_unlabeled=true` includes unlabeled split rows such as the public challenge `test` set.
- `dataset.source.class_id_file`, `train_labels_file`, `val_labels_file`, and `test_labels_file` let you override auto-discovered metadata files when the release uses custom filenames.

