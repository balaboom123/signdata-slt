# Contributing

## Project Structure

```
Sign-Language-Preprocessing/
├── configs/
│   ├── _base/                  # Shared base configs
│   │   ├── pose_mediapipe.yaml
│   │   ├── pose_mmpose.yaml
│   │   └── video.yaml
│   ├── youtube_asl/            # YouTube-ASL dataset configs
│   └── how2sign/               # How2Sign dataset configs
├── src/sign_prep/
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Argument parsing
│   ├── registry.py             # Component registry
│   ├── config/                 # YAML loading & Pydantic schema
│   ├── pipeline/               # PipelineRunner & PipelineContext
│   ├── datasets/               # Dataset definitions
│   ├── processors/             # Pipeline step implementations
│   ├── extractors/             # MediaPipe & MMPose extractors
│   ├── models/                 # MMPose model configs & checkpoints
│   └── utils/                  # Video, file, and text utilities
├── docs/                       # Documentation
├── assets/                     # Video ID lists, demo files
├── tests/                      # Test suite
└── requirements.txt
```

---

## Adding a New Dataset

1. Create a class in `src/sign_prep/datasets/` decorated with `@register_dataset`:

```python
from sign_prep.datasets.base import BaseDataset
from sign_prep.registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config):
        # Raise ValueError for invalid configs
        pass
```

2. Import it in `src/sign_prep/datasets/__init__.py` so the decorator runs at startup.

3. Create a config directory and YAML under `configs/my_dataset/`. See [configuration reference](docs/configuration.md#minimal-working-config) for the minimal required fields.

4. Add dataset documentation to `docs/datasets.md`.

---

## Adding a New Processor

1. Create a class in `src/sign_prep/processors/` decorated with `@register_processor`:

```python
from sign_prep.processors.base import BaseProcessor
from sign_prep.registry import register_processor

@register_processor("my_step")
class MyProcessor(BaseProcessor):
    name = "my_step"

    def run(self, context):
        # processing logic; return updated context
        return context

    def validate(self, context) -> bool:
        # optional pre-check
        return True
```

2. Import it in the appropriate `processors/__init__.py`.

3. Add `my_step` to the `pipeline.steps` list in your config YAML.

---

## Adding a New Extractor

1. Create a class in `src/sign_prep/extractors/` decorated with `@register_extractor`:

```python
from sign_prep.extractors.base import LandmarkExtractor
from sign_prep.registry import register_extractor

@register_extractor("my_extractor")
class MyExtractor(LandmarkExtractor):
    def process_frame(self, frame):
        # Return array of shape (K, 4) with [x, y, z, visibility], or None
        ...

    def close(self):
        # Release any resources
        ...
```

2. Import it in `src/sign_prep/extractors/__init__.py`.

3. Set `extractor.name: my_extractor` in your config YAML.

---

## Running Tests

```bash
pytest tests/
```

See [architecture docs](docs/architecture.md) for how the registry and pipeline fit together.
