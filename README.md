<!-- H1 -->
# SignDATA: Data Pipeline for Sign Language Translation

<!-- Animated Header -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=SignDATA&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Config-driven%20Pose/Video%20Preprocessing%20Pipeline&descAlignY=52&descSize=18" alt="SignDATA – Data Pipeline for Sign Language Translation"/>

<p align="center">
  <a href="https://arxiv.org/pdf/2604.20357"><img src="https://img.shields.io/badge/arXiv-2604.20357-b31b1b?style=flat" alt="arXiv"/></a> &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-10B981?style=flat" alt="License"/></a> &nbsp;
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat" alt="Python 3.11+"/></a>
</p>

SignDATA is a config-driven preprocessing pipeline for sign-language datasets.
It builds manifests, extracts pose landmarks or video representations, and
packages training samples as WebDataset shards.

## Installation

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv .venv
source .venv/bin/activate  # Linux/macOS; use .venv\Scripts\activate on Windows
python -m pip install -e .
python -m signdata run --list-presets
```

Use `python -m pip install -e ".[dev]"` for development and tests. MMPose and
MMDetection require the additional stack in the
[installation guide](docs/guides/installation.md).

## Quick Start

Run the shipped YouTube-ASL MediaPipe job:

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml
```

The job downloads available source material, builds a manifest, writes raw and
normalized landmark arrays under `dataset/youtube_asl/output/default/`, and
creates shards under `dataset/youtube_asl/webdataset/default/`. For a small,
offline input-to-shard smoke run with explicit success checks, use the
[verified quickstart](docs/guides/quickstart.md).

## Supported Datasets

| Dataset | Dataset | Dataset |
|---|---|---|
| [YouTube-ASL](docs/guides/datasets/youtube-asl.md) | [OpenASL](docs/guides/datasets/openasl.md) | [How2Sign](docs/guides/datasets/how2sign.md) |
| [BOBSL](docs/guides/datasets/bobsl.md) | [BSL-1K](docs/guides/datasets/bsl1k.md) | [WLASL](docs/guides/datasets/wlasl.md) |
| [MS-ASL](docs/guides/datasets/ms-asl.md) | [AUTSL](docs/guides/datasets/autsl.md) | [CSL](docs/guides/datasets/csl.md) |
| [LSA64](docs/guides/datasets/lsa64.md) | [SLoVo](docs/guides/datasets/slovo.md) | [RWTH-PHOENIX-Weather](docs/guides/datasets/rwth-phoenix-weather.md) |

The [dataset guide index](docs/guides/datasets/index.md) is authoritative for
setup requirements and shipped-job coverage.

## Documentation

Start with the [documentation index](docs/index.md). It routes installation,
dataset setup, configuration and CLI lookup, manifest/output contracts,
architecture, troubleshooting, and dated research notes.

## Citation

If you use SignDATA in your research, please cite:

```bibtex
@Article{chen2026signdata,
    author  = {Kuanwei Chen and Tingyi Lin},
    journal = {arXiv:2604.20357},
    title   = {SignDATA: Data Pipeline for Sign Language Translation},
    year    = {2026},
}
```

## License

The repository code and documentation are licensed under the [MIT License](LICENSE).
External datasets and research artifacts remain subject to their original terms.
