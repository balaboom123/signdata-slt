# Pipeline Model

The pipeline is a fixed-order state machine. A job config selects which stages
are active; it does not reorder them.

```text
job YAML
  │
  ▼
load_config()          resolve bases, paths, and CLI overrides
  │
  ▼
PipelineRunner(config)
  │  ├─ dataset.download       when dataset.download is true
  │  ├─ dataset.manifest       when dataset.manifest is true
  │  ├─ processing.<processor> when processing.enabled is true
  │  ├─ post_processing.normalize
  │  │                         when post_processing.enabled is true
  │  └─ output.webdataset      when output.enabled is true
  │
  ▼
PipelineContext (final)
```

If manifest generation is disabled but `paths.manifest` is set, the runner
loads and validates that existing manifest before processing. Available rows
are filtered immediately before downstream stages.

Experiment runs load `configs/experiments/*.yaml`, then execute each referenced
job under `configs/jobs/` with its own override set. Jobs are isolated by
`run_name` but follow the same fixed sequence.

## Context Threading

Every stage receives a `PipelineContext` and returns the context for the next
stage. The runner resolves paths once at startup; stages communicate through
these fields rather than recomputing global paths.

| Field | Type | Meaning |
|---|---|---|
| `config` | `Config` | Parsed job configuration |
| `dataset` | `DatasetAdapter` | Registered active adapter |
| `output_dir` | `Path?` | `{paths.output}/{run_name}` |
| `webdataset_dir` | `Path?` | `{paths.webdataset}/{run_name}` |
| `videos_dir` | `Path?` | Adapter-resolved source video directory |
| `manifest_path` | `Path?` | Current manifest path |
| `manifest_df` | `DataFrame?` | Current in-memory manifest |
| `force_all` | `bool` | Whether processors must overwrite existing outputs |
| `stats` | `dict[str, dict]` | Per-stage counters accumulated during the run |

Dataset adapters populate the manifest path and frame. Processors read that
frame and record their counters. Post-processing and output consume the same
context, so a stage boundary is explicit even though execution is in-process.

## Resume and Force Behavior

Processing resumes at the artifact level. `video2pose` and `video2crop` skip an
existing sample output; `video2parts` skips a sample only when all five expected
artifacts exist; `video2compression` recognizes both encoded and passthrough
mirror files. Normalization likewise skips existing normalized arrays.

`--force-all` sets `PipelineContext.force_all`, disabling those existence-based
skips for processing and normalization. Dataset acquisition and manifest
construction still follow their adapter-specific behavior, and WebDataset
shards are always opened for a new output run rather than treated as a general
stage checkpoint.

## Why the Order Is Fixed

Each boundary consumes state established by the previous boundary: processing
requires a manifest, normalization requires raw landmark arrays, and output
packages the active processor artifacts. Config flags can stop after any
boundary or start from an existing manifest, but arbitrary reordering would
break those contracts.

## See Also

- [Architecture](architecture.md) -- components, registries, and package ownership
- [Stage reference](../reference/stages.md) -- activation rules and stage I/O
- [Configuration reference](../reference/configuration.md) -- stage flags and paths
