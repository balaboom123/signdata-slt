# SignDATA Documentation

| I want to… | Read |
|---|---|
| install SignDATA or configure GPU backends | [Installation](guides/installation.md) |
| run a deterministic job from input video to a shard | [Quickstart](guides/quickstart.md) |
| prepare one of the supported datasets | [Dataset guides](guides/datasets/index.md) |
| diagnose environment or runtime failures | [Troubleshooting](guides/troubleshooting.md) |
| look up YAML fields and defaults | [Configuration](reference/configuration.md) |
| look up commands and flags | [CLI](reference/cli.md) |
| understand manifest columns | [Manifest](reference/manifest.md) |
| consume NPY, NPZ, MP4, JSON, or WebDataset outputs | [Output formats](reference/outputs.md) |
| check stage activation and paths | [Pipeline stages](reference/stages.md) |
| understand components and code ownership | [Architecture](concepts/architecture.md) |
| understand stage order, context, and resume behavior | [Pipeline model](concepts/pipeline-model.md) |
| read point-in-time research or audit records | [Notes](notes/index.md) |

## Guides

Admission rule: **the reader wants to accomplish X**. Guides contain ordered,
runnable steps. They are not exhaustive field references.

- [Installation](guides/installation.md)
- [Quickstart](guides/quickstart.md)
- [Troubleshooting](guides/troubleshooting.md)
- [Dataset guides](guides/datasets/index.md)

## Reference

Admission rule: **the reader needs to look up Y**. Reference pages describe
current facts exhaustively and are organized for scanning rather than
narrative reading.

- [Configuration](reference/configuration.md)
- [CLI](reference/cli.md)
- [Pipeline stages](reference/stages.md)
- [Manifest](reference/manifest.md)
- [Output formats](reference/outputs.md)

## Concepts

Admission rule: **the reader wants to understand why**. Concept pages explain
relationships and design choices without becoming task checklists.

- [Architecture](concepts/architecture.md)
- [Pipeline model](concepts/pipeline-model.md)

## Notes

Admission rule: **the content is a dated record**. Notes preserve point-in-time
research, experiments, and reports and are explicitly not maintained as current
truth.

- [Notes index and status](notes/index.md)

## Open Questions

The code does not establish answers for these contract decisions:

- Should `TEXT` become required, and should WLASL derive it from `GLOSS`?
- What timebase do the upstream RWTH-PHOENIX-Weather `start` and `end` fields
  use, and should the adapter synthesize seconds from each materialized clip
  when those fields are absent?
- Which exact columns should SignDATA require from the externally supplied
  How2Sign manifest after legacy-name normalization?
- Should `video2parts` read `SOURCE_URL` (emitted by WLASL and MS-ASL) in
  addition to its current `URL` lookup for `meta.json.source_url`?
- Should the How2Sign `video2parts` metadata fields be admitted by
  `How2SignSourceConfig`, or should they move out of `dataset.source`? The
  shipped parts job supplies them, but the adapter currently forbids them.
