# CLI Reference

Use `python -m signdata` as the canonical form. An editable install also exposes
the equivalent `signdata` console script.

<!-- BEGIN GENERATED: cli -->
## Commands

| Command | Description |
|---|---|
| `run` | Run a preprocessing job |
| `experiment` | Run a multi-job experiment |

## `run`

```text
usage: python -m signdata run [-h] [--force-all] [--run-name NAME]
                              [--list-presets] [--override [KEY=VALUE ...]]
                              [CONFIG]
```

| Argument | Required | Default | Description |
|---|---:|---|---|
| `-h, --help` | no | — | show this help message and exit |
| `[CONFIG]` | no | `null` | Path to YAML job config file (required unless --list-presets is used) |
| `--force-all` | no | `false` | Force rerun of all stages |
| `--run-name NAME` | no | `null` | Override run_name for path isolation |
| `--list-presets` | no | `false` | List available keypoint presets and exit |
| `--override [KEY=VALUE ...]` | no | `[]` | Config overrides: key=value (e.g. processing.max_workers=8) |

## `experiment`

```text
usage: python -m signdata experiment [-h] [--force-all] CONFIG
```

| Argument | Required | Default | Description |
|---|---:|---|---|
| `-h, --help` | no | — | show this help message and exit |
| `CONFIG` | yes | — | Path to YAML experiment config file |
| `--force-all` | no | `false` | Force rerun of all stages in every job |

## Keypoint Presets

| Name | Description |
|---|---|
| `mediapipe_543_to_83` | MediaPipe unrefined: 6 pose + 35 face + 21 left hand + 21 right hand |
| `mediapipe_553_to_85` | MediaPipe refined: 6 pose + 37 face + 21 left hand + 21 right hand |
| `mmpose_133_to_85` | COCO WholeBody 133 to 85 selected upper body, face, and hand keypoints |
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
