<!-- H1 -->
# SignDATA: Data Pipeline for Sign Language Translation

<!-- Animated Header -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=SignDATA&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Config-driven%20Pose/Video%20Preprocessing%20Pipeline&descAlignY=52&descSize=18" alt="SignDATA – Data Pipeline for Sign Language Translation"/>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-10B981?style=flat" alt="License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat" alt="Python 3.11+"/></a>
  <a href="https://github.com/balaboom123/signdata-slt/stargazers"><img src="https://img.shields.io/github/stars/balaboom123/signdata-slt?style=flat&color=F59E0B" alt="Stars"/></a>
  <a href="https://github.com/balaboom123/signdata-slt/issues"><img src="https://img.shields.io/github/issues/balaboom123/signdata-slt?style=flat&color=EF4444" alt="Issues"/></a>
</p>

A config-driven, modular pipeline for preprocessing multiple **Sign Language (SL)** datasets.
Supports two landmark extractors (**MediaPipe Holistic** and **MMPose RTMPose3D**) and two output modes (pose landmarks and video clips).

<!-- Quick Links -->
<div align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/🚀_Quick_Start-4285F4?style=flat-square" alt="Quick Start"/></a>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/📖_Installation-34A853?style=flat-square" alt="Installation"/></a>
  <a href="docs/pipeline-stages.md"><img src="https://img.shields.io/badge/🛠️_Pipeline_Stages-EA4335?style=flat-square" alt="Pipeline Stages"/></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/🏗️_Architecture-FBBC05?style=flat-square" alt="Architecture"/></a>
</div>

<br/>

---

## ✨ Key Features

<div align="center">
<table>
<tr>
<td width="50%">

### 📝 Config-Driven
YAML configs with base inheritance and CLI overrides

### 🦴 Two Extractors
MediaPipe Holistic (553 keypoints) and MMPose RTMPose3D (133 keypoints)

### 🎬 Two Pipeline Modes
`pose` (landmarks) and `video` (clip extraction)

</td>
<td width="50%">

### 🧩 Registry Architecture
Add datasets, processors, and extractors via decorators

### ⚡ Parallel Processing
Multi-worker extraction, normalization, and clipping

### 📦 WebDataset Output
Sharded tar archives for efficient training data loading

</td>
</tr>
</table>
</div>

> 📖 **New?** See the [Installation Guide](docs/installation.md) to get started.

---

## Installation

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional: MMPose (GPU required)

MediaPipe works on CPU out of the box. MMPose requires a CUDA-capable GPU and additional dependencies -- see the [Installation Guide](docs/installation.md) for full setup instructions.

---

## Quick Start

```bash
# Download YouTube-ASL videos, extract MediaPipe landmarks, normalize, and package into WebDataset shards
python -m signdata configs/youtube_asl/pose_mediapipe.yaml

# Extract MMPose landmarks from pre-downloaded How2Sign data (CUDA required)
python -m signdata configs/how2sign/pose_mmpose.yaml

# Override any config value from the command line (e.g. more workers, stop after extraction)
python -m signdata configs/youtube_asl/pose_mediapipe.yaml \
  --override processing.max_workers=8 pipeline.stop_at=extract
```

---

## Output

Both modes produce [WebDataset](https://github.com/webdataset/webdataset) tar shards for efficient training data loading. See [Pipeline Stages](docs/pipeline-stages.md) for detailed output formats and data shapes.

---

## Supported Datasets

| Dataset | Venue | Description | License |
|:--------|:------|:------------|:--------|
| **[YouTube-ASL](docs/datasets.md#youtube-asl)** | NeurIPS 2023 | 11,000+ videos, 73,000+ segments -- open-domain ASL-English parallel corpus | [Apache-2.0](https://github.com/google-research/google-research/tree/master/youtube_asl) |
| **[How2Sign](docs/datasets.md#how2sign)** | CVPR 2021 | 80+ hours of instructional ASL in a controlled studio environment | [CC BY-NC 4.0](https://how2sign.github.io/) |

For paper-aligned preprocessing methodology, see [Research-Aligned Preprocessing](docs/research-preprocessing.md).

---

## Documentation

- [Installation Guide](docs/installation.md) -- base setup and MMPose GPU dependencies
- [Architecture](docs/architecture.md) -- system design, registry, pipeline flow
- [Configuration](docs/configuration.md) -- full config reference, inheritance, CLI overrides
- [Pipeline Stages](docs/pipeline-stages.md) -- all 6 processing stages
- [Datasets](docs/datasets.md) -- YouTube-ASL vs How2Sign setup
- [Research-Aligned Preprocessing](docs/research-preprocessing.md) -- paper-aligned preprocessing notes

## License

The MIT license in this repository applies to the code and documentation in this project. Use of external datasets, research artifacts, and upstream repos referenced above must comply with their original licenses and usage terms.

MIT -- see [LICENSE](LICENSE).
