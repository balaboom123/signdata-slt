# Architecture

## Entry Points

```bash
python -m signdata run <job.yaml> [--override key=value ...]
python -m signdata experiment <experiment.yaml>
```

`src/signdata/__main__.py` imports `signdata.datasets` and
`signdata.processors` to populate registries before any config is executed.

## Execution Model

The fixed stage order, context threading, and resume behavior are documented in [Pipeline Model](pipeline-model.md).

## Registries

Two global registries live in `src/signdata/registry.py`:

| Decorator | Registry | Base class |
|---|---|---|
| `@register_dataset(name)` | `DATASET_REGISTRY` | `DatasetAdapter` |
| `@register_processor(name)` | `PROCESSOR_REGISTRY` | `BaseProcessor` |

## Processing

The pipeline runner dispatches to the processor specified by
`config.processing.processor`:

- `video2pose` — video → pose landmarks (.npy), using detection + pose backends
- `video2crop` — video → cropped video (.mp4), using detection + `src/signdata/processors/video/ffmpeg.py`
- `video2compression` — source-video mirror with codec-aware passthrough or re-encoding
- `video2parts` — video → face/hand crop streams plus upper-body pose, using MediaPipe Holistic

Video processors share helpers in `processors/video/ffmpeg.py`.
`video2pose`, `video2crop`, and `video2parts` decode manifest segments;
`video2compression` probes and transcodes complete source videos. Pose and parts
processing use the OpenCV fallback when FFmpeg is absent, while crop and
compression outputs require FFmpeg.

When enabled, post-processing runs landmark normalization.

## Package Layout

- `src/signdata/config/` contains Pydantic config schemas (`schema.py`), YAML loading and path resolution (`loader.py`), and experiment config parsing (`experiment.py`).
- `src/signdata/pipeline/` contains the pipeline runner (`runner.py`), shared pipeline context (`context.py`), and experiment runner (`experiment.py`).
- `src/signdata/datasets/` contains dataset adapter packages.
- `src/signdata/datasets/_ingestion/` contains dataset-ingestion helpers used only during `dataset.download` and `dataset.manifest`. Must NOT be imported by processors, pipeline runner, or output modules — use `signdata.utils` for those.
- `src/signdata/processors/detection/` contains detector backends and bbox utilities.
- `src/signdata/processors/pose/` contains pose estimators and presets.
- `src/signdata/processors/video/` contains shared video helpers such as `ffmpeg.py`.
- `src/signdata/processors/` contains top-level processors such as `video2pose` and `video2crop`.
- `src/signdata/post_processors/` contains landmark normalization.
- `src/signdata/output/` contains output writers such as `webdataset`.
- `src/signdata/utils/` contains pipeline-wide helpers for video I/O, file discovery, manifest reading/validation, and other generic logic.
- `resources/` contains shipped model config assets.

## Contributor Boundary Rule

When adding new shared code, use this table to decide where it goes:

| Directory | Purpose | Consumers |
|---|---|---|
| `datasets/_ingestion/` | Acquire raw external data, build canonical manifests | Dataset adapter `source.py` and `manifest.py` only |
| `utils/` | Operate on already-built manifests or video files during processing | Processors, pipeline runner, output modules |

**Rule of thumb:** if the code runs once when setting up a dataset → `_ingestion/`. If it runs per-sample during the processing pipeline → `utils/`.

## Dataset Adapter Structure

All datasets must be packages now. The default structure is:

```text
src/signdata/datasets/
├── _ingestion/
│   ├── availability.py
│   ├── classmap.py
│   ├── media.py
│   ├── text.py
│   └── youtube.py
└── <dataset_name>/
    ├── __init__.py
    ├── adapter.py
    ├── source.py
    └── manifest.py
```

- `adapter.py` registers the dataset and keeps stage methods thin.
- `source.py` owns `SourceConfig`, path resolution, validation, download, and preparation/materialization.
- `manifest.py` owns source parsing and canonical manifest construction.

Add more files only when the dataset is large enough to justify them. Typical
extensions are `schema.py`, `constants.py`, `parsing.py`, or `splits.py`.

## See Also

- [Pipeline model](pipeline-model.md) -- execution order, context, and resume behavior
- [Configuration Reference](../reference/configuration.md) -- config layout and key fields
- [Pipeline Stages](../reference/stages.md) -- stage-by-stage behavior
- [Datasets](../guides/datasets/index.md) -- dataset-specific setup notes
