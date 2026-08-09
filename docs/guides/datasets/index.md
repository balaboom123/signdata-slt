# Dataset Guides

Admission rule: these pages are task-oriented. Add a page here when a reader wants ordered steps to set up or process one dataset.

| Dataset | Guide | Shipped job |
|---|---|---|
| YouTube-ASL | [Setup and processing](youtube-asl.md) | Pose and video |
| OpenASL | [Setup and manifest](openasl.md) | No complete job |
| How2Sign | [Setup and processing](how2sign.md) | Pose, parts, and video |
| BOBSL | [Setup and processing](bobsl.md) | Pose and video |
| BSL-1K | [Setup and processing](bsl1k.md) | Pose and video |
| WLASL | [Setup and processing](wlasl.md) | Pose and video |
| MS-ASL | [Setup and processing](ms-asl.md) | Pose and video |
| AUTSL | [Setup and processing](autsl.md) | Pose and video |
| CSL | [Setup and processing](csl.md) | Pose and video |
| LSA64 | [Setup and processing](lsa64.md) | Pose and video |
| SLoVo | [Setup and manifest](slovo.md) | No complete job |
| RWTH-PHOENIX-Weather | [Setup and processing](rwth-phoenix-weather.md) | Pose and video |

## Adding a New Dataset

All datasets use `src/signdata/datasets/<dataset_name>/` with `adapter.py`, `source.py`, and `manifest.py` as the default entry files.

See [CONTRIBUTING.md](../../../CONTRIBUTING.md#adding-a-new-dataset) for the required structure, responsibilities, and code template.

## See Also

- [Pipeline stages](../../reference/stages.md) -- stage activation and I/O
- [Configuration reference](../../reference/configuration.md) -- config schema and overrides
- [Research notes](../../notes/index.md) -- dated paper-aligned records
