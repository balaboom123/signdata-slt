# Architecture

## Entry Point

```bash
python -m signdata <config.yaml> [--override key=value ...]
```

`__main__.py` imports all registry modules, loads the YAML config, and hands it to `PipelineRunner`.

## Pipeline Flow

```
YAML config
  │
  ▼
load_config()          # merge _base → dataset YAML → CLI overrides
  │
  ▼
PipelineRunner(config)
  │  ├─ look up dataset in DATASET_REGISTRY
  │  └─ build processor chain from pipeline.steps
  │
  ▼
for each processor:
  │  processor.validate(context)
  │  context = processor.run(context)
  │  context.completed_steps.append(name)
  ▼
PipelineContext (final)
```

`PipelineContext` is a dataclass that carries shared state between steps: the config, dataset instance, project root, manifest path/DataFrame, completed steps, and per-step stats.

## Registry System

Three global registries in `registry.py`, populated via decorators:

| Decorator | Registry | Base class |
|---|---|---|
| `@register_dataset(name)` | `DATASET_REGISTRY` | `BaseDataset` |
| `@register_processor(name)` | `PROCESSOR_REGISTRY` | `BaseProcessor` |
| `@register_extractor(name)` | `EXTRACTOR_REGISTRY` | `LandmarkExtractor` |

Registration happens at import time. `__main__.py` imports `signdata.datasets`, `signdata.processors`, and `signdata.extractors` to trigger it.

## Pipeline Modes

**`pose` mode** (landmarks):
`download → manifest → extract → normalize → webdataset`
Extracts per-frame pose landmarks as `.npy` arrays, normalizes them, and packages into tar shards.

**`video` mode** (clips):
`download → manifest → clip_video → webdataset`
Clips video segments with ffmpeg and packages `.mp4` files into tar shards.

The `steps` list in config controls exactly which processors run. `start_from` and `stop_at` allow resuming or stopping partway through.

### Data Shape Flow

| Stage | Format | Shape / Notes |
|---|---|---|
| `extract` output | `.npy` per segment | `(T, K, 4)` — T frames, K keypoints, 4 channels `[x, y, z, vis]` |
| `normalize` output | `.npy` per segment | `(T, K'×C)` flattened — K' reduced keypoints, C = 3 (visibility is stripped); C = 2 when `remove_z=true` |
| `clip_video` output | `.mp4` per segment | raw video clip, optionally resized |
| `webdataset` output | `.tar` shards | each sample: `.npy`/`.mp4` + `.txt` + `.json` metadata |

Default keypoint counts (K): MediaPipe refined = **553**, MediaPipe unrefined = **543**, MMPose RTMPose3D = **133**.
Default after reduction (K'): **85** keypoints for MediaPipe refined and MMPose; **83** for MediaPipe unrefined. Dataset-specific configs may override via `normalize.keypoint_indices`.

## PipelineContext Fields

`PipelineContext` is a dataclass that carries all shared state between steps:

| Field | Type | Set by | Description |
|---|---|---|---|
| `config` | `Config` | runner | Full parsed config object |
| `dataset` | `BaseDataset` | runner | Dataset instance (e.g. `YouTubeASL`) |
| `project_root` | `Path` | runner | Absolute path to repo root |
| `manifest_path` | `Path?` | `manifest` processor | Path to the manifest CSV |
| `manifest_df` | `DataFrame?` | `manifest` processor | Loaded manifest as a pandas DataFrame |
| `completed_steps` | `list[str]` | each processor | Names of processors that have finished |
| `stats` | `dict[str, dict]` | each processor | Per-step counters (processed, skipped, errors) |

## Processors

Every processor subclasses `BaseProcessor` and implements:

- `run(context) → context` -- execute the step, return updated context
- `validate(context) → bool` -- optional pre-check (default: `True`)

Processors are stateless between runs. All shared state flows through `PipelineContext`.

## Extensibility

The pipeline is designed to be extended with new datasets, processors, and extractors via the registry decorator pattern. See [CONTRIBUTING.md](../CONTRIBUTING.md) for step-by-step instructions and code examples.

---

## See Also

- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Pipeline Stages](pipeline-stages.md) -- detailed per-stage documentation
- [Datasets](datasets.md) -- dataset-specific setup guides
