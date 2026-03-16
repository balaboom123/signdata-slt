# YouTube-ASL Research Preprocessing

This note summarizes the preprocessing method described in the YouTube-ASL paper:

- Uthus, Tanzer, Georg. "YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus"

The goal here is to explain how the paper prepares data before training, and how that maps to the experiment config in this repository.

## 1. Corpus Construction Before Training

The paper builds the dataset in two stages:

1. Automatic retrieval of candidate YouTube videos
2. Human filtering of those candidates by skilled ASL annotators

### Automatic retrieval

The candidate pool is collected from public YouTube videos tagged as related to sign language or American Sign Language. The paper then narrows this set by applying several heuristic filters before annotation:

- Keep videos with user-provided captions rather than speech-derived automatic captions.
- Filter by video quality constraints such as duration, resolution, and frame rate.
- Exclude videos where none of the caption spans contain exactly one visible person.

This produces 88,002 candidate videos.

### Human filtering

The paper then uses 3 native ASL signers with English proficiency to review the candidate videos. Videos are kept only when the captions are generally good English translations of the signed ASL and the overall alignment quality is acceptable.

This filtering step reduces the dataset to:

- 11,093 videos
- 984 total hours of video
- 610,193 captions

## 2. Training-Time Preprocessing in the Paper

The paper's baseline model trains on single-caption clips rather than full videos.

### English target preprocessing

For the English side, the paper uses the raw captions as targets and clips each training example to the boundary of a single caption.

The paper filters out caption segments when:

- Caption length is greater than 300 characters
- Segment duration is less than 0.2 seconds
- Segment duration is greater than 60 seconds
- The corresponding video span does not contain exactly one person

The paper explicitly states that it does not lowercase the text or apply additional text normalization.

### Sign input preprocessing

For the sign side, the paper does not use raw RGB frames as model input. Instead, it extracts pose-style features with MediaPipe Holistic.

MediaPipe Holistic predicts landmarks for:

- Pose
- Face
- Left hand
- Right hand

The paper then reduces the full landmark output to 85 selected keypoints using sign-language domain knowledge:

- 21 keypoints for the left hand
- 21 keypoints for the right hand
- 6 body keypoints: shoulders, elbows, and hips
- 37 face keypoints covering the eyes, eyebrows, lips, and face outline

This 85-keypoint subset is the critical part of the method. After reduction:

- Each keypoint keeps x, y, and z coordinates
- MediaPipe visibility values are ignored
- Missing landmarks are represented with a large negative sentinel value
- The landmarks are normalized over the whole clip so they fit inside a unit bounding box
- Every second frame is discarded to reduce sequence length

The final model input is therefore a half-frame-rate sequence of 255-dimensional vectors:

- 85 keypoints x 3 coordinates = 255 values per frame

Because the paper drops every second frame instead of forcing a fixed target FPS, the effective frame rate varies with the source video. The paper notes that most examples end up around 12 to 15 FPS after this reduction.

## 3. Model Context

After preprocessing, the paper feeds the landmark sequence into a modified T5.1.1-Base model:

- A linear projection maps each 255-dimensional landmark frame into the T5 encoder space
- The encoder context window is 256 frames
- The decoder context window is 128 tokens

The important point for this repository is that the paper's baseline depends on landmark preprocessing choices at least as much as on the model itself.

## 4. Mapping to This Repository

The closest experiment config in this repo is:

- [`configs/exp/youtube_asl.yaml`](../../configs/exp/youtube_asl.yaml)

That config is aligned with the paper where the current config system exposes a setting:

- It inherits from `configs/_base/pose_mediapipe.yaml`
- It keeps the 85-keypoint landmark subset
- It uses `target_fps: null` with `frame_skip: 2` to match "discard every second frame"
- It uses `accept_fps_range: [15.0, 60.0]`
- It keeps `unvisible_value: -999.0` for missing landmarks
- It sets `mask_landmark_level: true` and `visibility_threshold: 0.0` so missing all-zero landmarks are masked while MediaPipe visibility scores are otherwise ignored

## 5. Current Gaps Between Paper and Repo

Some parts of the paper are not fully controlled by YAML config in the current codebase:

- The paper says to use raw captions without extra normalization, but the current manifest processor still normalizes transcript text.
- The paper filters examples to spans with exactly one visible person, but that constraint is not currently implemented as a dedicated config-controlled preprocessing step.

So the experiment config captures the paper-aligned landmark and sampling settings, while a few text and person-filtering details would still require code changes if exact reproduction is needed.

## Citation

```bibtex
@misc{uthus2023youtubeasl,
  author        = {Uthus, David and Tanzer, Garrett and Georg, Manfred},
  title         = {YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus},
  year          = {2023},
  eprint        = {2306.15162},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2306.15162},
}
```
