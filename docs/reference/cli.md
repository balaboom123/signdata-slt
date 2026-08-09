# CLI Reference

Use `python -m signdata` as the canonical form. An editable install also exposes
the equivalent `signdata` console script.

<!-- BEGIN GENERATED: cli -->
## `run`

```text
python -m signdata run [config] [--force-all] [--run-name NAME]
                                [--list-presets]
                                [--override [KEY=VALUE ...]]
```

| Argument | Required | Meaning |
|---|---:|---|
| `config` | Unless `--list-presets` is used | YAML job config path |
| `--force-all` | No | Rerun processing and normalization artifacts that already exist |
| `--run-name NAME` | No | Override `run_name` for isolated output directories |
| `--list-presets` | No | Print built-in keypoint presets and exit |
| `--override KEY=VALUE ...` | No | Apply one or more dot-separated config overrides |

## `experiment`

```text
python -m signdata experiment config [--force-all]
```

| Argument | Required | Meaning |
|---|---:|---|
| `config` | Yes | YAML experiment config path |
| `--force-all` | No | Force all stages in every experiment job |
<!-- END GENERATED: cli -->

Override values are parsed as YAML scalars and collections. Put `--override`
after other run flags because it accepts zero or more values:

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --run-name trial \
  --override processing.max_workers=8 output.max_shard_count=2000
```

To pass the literal string `null` rather than YAML null, preserve YAML quotes in
the shell argument:

```bash
python -m signdata run configs/jobs/how2sign/video.yaml \
  --override 'processing.detection="null"'
```

## See Also

- [Configuration reference](configuration.md) -- fields accepted by overrides
- [Pipeline model](../concepts/pipeline-model.md) -- force and resume behavior
- [Quickstart](../guides/quickstart.md) -- complete runnable example
