# Towards Privacy-Aware Sign Language Translation at Scale Research Preprocessing

This note summarizes the data preprocessing described in:

- Rust, Shi, Wang, Camgoz, and Maillard. "Towards Privacy-Aware Sign Language Translation at Scale"

The paper introduces a two-stage sign language translation pipeline, SSVP-SLT:

1. Stage I: self-supervised pretraining on anonymized sign-language video
2. Stage II: supervised SLT finetuning on a smaller parallel dataset

The important preprocessing shift compared with the existing landmark-based papers in this repo is that this work trains from RGB video, not pose landmarks, and it treats anonymization as part of data preparation rather than as a model-side detail.

## 1. High-Level Data Preparation

The paper uses three ASL datasets for different purposes:

- YouTube-ASL as the large-scale, in-the-wild source for pretraining and finetuning
- How2Sign as the smaller curated parallel dataset for finetuning and evaluation
- DailyMoth-70h for controlled experiments on the effect of face blurring

The general preprocessing recipe is:

1. Start from sign-language video in RGB form
2. Apply facial obfuscation so the pretraining data is anonymized
3. Train the video encoder on anonymized video
4. Finetune on a smaller caption-aligned translation dataset

In other words, the paper does not replace the signer with pose landmarks. It keeps the full visual signal and removes identity-sensitive facial detail by blurring faces before training.

## 2. What Counts as Preprocessing in This Paper

For this work, preprocessing has two distinct layers.

### Privacy preprocessing

The privacy-aware part is explicit:

- Faces are blurred before training
- The anonymized data is used in self-supervised video pretraining
- The smaller finetuning dataset can optionally remain unblurred if explicit consent exists, but the main reported How2Sign experiments in the paper use blurred data

This is the key difference from the pose-based configs already in this repository. The paper keeps RGB frames and changes the facial region, whereas the existing pose configs discard the raw image and keep only landmark coordinates.

### Video preparation for model input

After anonymization, the model consumes RGB clips with the following training-time assumptions:

- SignHiera uses clips of 128 frames
- Frames are sampled with temporal stride 2
- Input resolution is 224 x 224
- Videos shorter than 128 frames are padded
- Longer videos are handled with sliding windows when features are extracted for SLT

The paper also applies video augmentation during pretraining:

- Random cropping
- Horizontal flipping
- RandAug

These are training-time loader and augmentation decisions, not clip-generation steps in this preprocessing repo.

## 3. Mapping the Paper to This Repository

The closest mapping in this repo is to use the `video` pipeline mode and prepare caption-aligned RGB clips that can later be consumed by MAE or SLT training code.

The experiment config is:

- [`configs/exp/privacy_aware_slt_video.yaml`](../../configs/exp/privacy_aware_slt_video.yaml)

This config intentionally does the following:

- Switch from pose extraction to `video` mode
- Use the default `paths.videos` directory — place your pre-blurred videos there (the pipeline has no built-in face-blurring processor)
- Keep all frames during preprocessing by setting `target_fps: null` and `frame_skip: 1`
- Re-encode clipped segments with `libx264` so sentence boundaries are preserved more faithfully than keyframe-only copy clipping
- Package the resulting clips as WebDataset shards for downstream training

The YouTube-ASL experiment config still includes the manifest stage because the repo needs transcript timing information to cut sentence-level clips. The How2Sign config assumes the aligned CSV already exists and only clips plus shards the video.

## 4. Practical Interpretation of Stage I and Stage II

Within this repository, the paper's two stages map to preprocessing as follows:

- Stage I pretraining: generate anonymized RGB clips and ignore the text when training the video encoder
- Stage II finetuning: reuse the same clip inventory together with the caption text from the manifest

This means the preprocessing artifact can stay the same while the downstream training code decides whether text supervision is used.

## 5. Current Gaps Between the Paper and Repo

Some paper details are not directly implemented as config-driven preprocessing in the current codebase:

- There is no built-in face-blurring processor, so videos must be anonymized before running these configs
- There is no config-controlled 128-frame window sampler; the paper's `128 x 2` sampling happens during model training
- There is no config-controlled random crop, horizontal flip, or RandAug stage for RGB clips
- There is no dataset config for DailyMoth-70h in the current repo

So the new experiment YAMLs capture the paper's preprocessing direction correctly: RGB clips instead of landmarks, privacy-aware blurred inputs, and clip packaging for downstream training. Exact reproduction of the full SSVP-SLT training recipe would still require training-side code outside this preprocessing pipeline.

## Citation

```bibtex
@misc{rust2024towardsprivacyawaresignlanguage,
  title         = {Towards Privacy-Aware Sign Language Translation at Scale},
  author        = {Phillip Rust and Hanyu Shi and Shuo Wang and Necati Cihan Camgoz and Jean Maillard},
  year          = {2024},
  eprint        = {2402.09611},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2402.09611},
}
```
