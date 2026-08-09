# Contributing

## Source Layout

The authoritative package and dataset-adapter layout lives in the
[architecture reference](docs/concepts/architecture.md#package-layout). Keep
structural descriptions there instead of copying the tree into contributor or
configuration pages.

## Adding a New Dataset

All datasets must be packages now. Do not add new flat modules like
`src/signdata/datasets/my_dataset.py`.

Use this default layout:

```text
src/signdata/datasets/
├── _ingestion/               # Dataset-ingestion helpers only
│   ├── availability.py
│   ├── classmap.py
│   ├── media.py
│   ├── text.py
│   └── youtube.py
└── my_dataset/
    ├── __init__.py
    ├── adapter.py
    ├── source.py
    └── manifest.py
```

### File Responsibilities

- `adapter.py`
  - Public dataset entrypoint only.
  - Contains `@register_dataset("my_dataset")` and the adapter class.
  - Keeps `download()` and `build_manifest()` thin by delegating to `source.py` and `manifest.py`.
- `source.py`
  - Owns `SourceConfig`, source path resolution, release discovery, validation, download, and preparation/materialization.
- `manifest.py`
  - Owns source metadata parsing and canonical manifest construction.
  - Handles split assignment, class-map joins, timing/bbox conversion, and TSV writing.

Add more files only when needed:

- `schema.py` for multiple typed row/config models
- `constants.py` for larger aliases or bundled filename tables
- `parsing.py` or `splits.py` when `manifest.py` becomes too dense

### Shared Helper Rule

- If a helper is used only by dataset ingestion or manifest building, put it in `src/signdata/datasets/_ingestion/`.
- If a helper is generic to the pipeline, processors, or outputs, keep it in `src/signdata/utils/`.

### Minimal Template

`src/signdata/datasets/my_dataset/__init__.py`

```python
from .adapter import MyDataset
```

`src/signdata/datasets/my_dataset/adapter.py`

```python
from pathlib import Path

from signdata.datasets.base import DatasetAdapter
from signdata.registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("my_dataset")
class MyDataset(DatasetAdapter):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config) -> None:
        pass

    def download(self, config, context):
        source = _source.get_source_config(config)
        context.stats["dataset.download"] = _source.validate(source, config, self.logger)
        return context

    def build_manifest(self, config, context):
        source = _source.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        return context
```

### Integration Steps

1. Create `src/signdata/datasets/my_dataset/` using the package layout above.
2. Export the adapter in `src/signdata/datasets/my_dataset/__init__.py`.
3. Re-export it from `src/signdata/datasets/__init__.py`.
4. Add a job YAML under `configs/jobs/my_dataset/`, for example `configs/jobs/my_dataset/mediapipe.yaml`. See [configuration reference](docs/reference/configuration.md#minimal-working-config).
5. Add dataset documentation to `docs/guides/datasets/<dataset-name>.md`.

## Adding a New Processor

1. Create a processor module under `src/signdata/processors/` and decorate its
   `BaseProcessor` subclass with `@register_processor`:

```python
from signdata.processors.base import BaseProcessor
from signdata.registry import register_processor


@register_processor("my_step")
class MyProcessor(BaseProcessor):
    name = "my_step"

    def run(self, context):
        return context
```

2. Import the module from `src/signdata/processors/__init__.py`; that import
   populates `PROCESSOR_REGISTRY` before the runner starts.
3. Add the registry key to the `ProcessingConfig.processor` `Literal` in
   `src/signdata/config/schema.py`. The four-stage runner dispatches through
   `PROCESSOR_REGISTRY`; there is no separate recipe file.
4. Add schema, processor, and config tests plus at least one runnable job or a
   clear explanation of why the processor is library-only.

## Adding a Pose Backend

Pose backends use an explicit factory, not a registry:

1. Add a `LandmarkExtractor` implementation under
   `src/signdata/processors/pose/<backend>/` and export it from that package.
2. Add a typed backend config model to `src/signdata/config/schema.py`, include
   it in `POSE_CONFIG_MAP`, and add the backend name to the
   `ProcessingConfig.pose` `Literal`.
3. Add the construction branch to `create_estimator()` in
   `src/signdata/processors/pose/base.py`.
4. Add unit tests, a runnable job where appropriate, and regenerate the config
   reference.

`src/signdata/registry.py` defines only dataset and processor registries. Job
configs select a backend with `processing.pose` and configure it with
`processing.pose_config`.

## Documentation

Put each page in exactly one purpose-based folder:

| Location | Admission rule |
|---|---|
| `docs/guides/` | The reader wants to accomplish a task; give ordered, runnable steps. |
| `docs/reference/` | The reader needs to look up facts; be exhaustive and scannable. |
| `docs/concepts/` | The reader wants to understand why; explain the model without task steps. |
| `docs/notes/` | The content is a dated record; mark its status and do not maintain it as current truth. |

Link to the authoritative page when a fact already has a home. Do not copy
package layouts, field defaults, or other reference facts into task guides.

When `src/signdata/config/schema.py` or `src/signdata/cli.py` changes,
regenerate the code-derived tables and commit the result:

```bash
python scripts/gen_docs_reference.py
```

The generator fails when a configured marker is absent or malformed. Run it
twice when changing the generator itself; the second run must leave both
reference files unchanged.

Run the fast documentation guards while editing Markdown:

```bash
python -m pytest -q tests/docs
```

They check local links and anchors, generated-reference freshness, config paths
and override keys in bash blocks, `dataset.source` table coverage, and leaked
machine-local home paths.

## Running Tests

```bash
python -m pytest -q tests
```

`configs/test/` is gitignored, but the integration config-schema test globs it
when it exists. Remove stale local files there or keep them valid; otherwise a
checkout can fail tests in a way CI cannot reproduce.

See [architecture docs](docs/concepts/architecture.md) for how the registries
and pipeline runner fit together.
