# Manifest Reference

The manifest is the tabular contract between dataset adapters and every
processing, post-processing, and output stage. Built-in writers use tab-separated
files even when a path ends in `.csv`.

There is currently no model or validator for the manifest as a whole. Each
adapter constructs a `pandas.DataFrame`, and generic consumers fail only when a
column they use is missing. The tables below describe current code, not an
enforced schema.

## Processing Contract

These four columns are required for every manifest that reaches processing or
WebDataset output:

| Column | Meaning | Consumer behavior |
|---|---|---|
| `SAMPLE_ID` | Stable identifier for one segment or isolated clip; intended to be unique within a manifest | Names sample artifacts and the WebDataset `__key__` |
| `VIDEO_ID` | Source video stem or logical source identifier | Used when `REL_PATH` is absent and recorded in output metadata |
| `START` | Segment start in seconds from the source-video origin | Passed to FFmpeg/OpenCV decoding |
| `END` | Segment end in seconds from the source-video origin | Consumers decode `END - START` seconds |

`get_timing_columns()` accepts only `START` and `END`. It does not accept frame
indices. The loader does normalize legacy How2Sign names
`START_REALIGNED`/`END_REALIGNED` to these names, after which downstream code
still interprets the values as seconds.

No generic function currently verifies uniqueness, non-empty identifiers,
`START >= 0`, or `END > START`. Adapter writers normally establish those
properties themselves.

## Common Optional Columns

| Column | Meaning | Behavior when absent |
|---|---|---|
| `TEXT` | Spoken-language target or a gloss copied into text form | WebDataset writes an empty `.txt`; processing is unaffected |
| `SPLIT` | Dataset-defined partition such as `train`, `val`, `dev`, `test`, or `all` | Generic processing is unaffected; the split is not automatically preserved in WebDataset metadata |
| `REL_PATH` | Video path relative to `paths.videos`, including extension | Resolution falls back to legacy `VIDEO_NAME`, then `VIDEO_ID` plus a known video extension |
| `GLOSS` | Sign-language lexical label or gloss sequence | No generic consumer uses it |
| `CLASS_ID` | Dataset-local lexical class identifier | No generic consumer uses it |
| `SIGNER_ID` | Dataset-local signer identifier | No generic consumer uses it |
| `FPS` | Source or adapter-assumed frames per second | Processing probes the video independently |
| `AVAILABLE` | Boolean written only by `availability_policy: mark_unavailable` | The runner drops false rows immediately before processing |
| `PERSON_DETECTED` | Whether source annotations supplied a person box | Generic processors do not use it |
| `BBOX_X1`…`BBOX_Y2` | Source-provided box coordinates; coordinate convention remains dataset-specific | Generic processors run their configured detector instead |

`REL_PATH` is the preferred locator. If the exact extension does not exist,
resolution retries the same stem across `.mp4`, `.webm`, `.mkv`, `.avi`, and
`.mov`. This makes a `video2compression` mirror usable without rewriting the
manifest.

## Coverage Matrix

`✓` means the built-in writer always creates the column, `○` means the field is
conditional, view-dependent, or delegated to an input manifest, and `—` means
the writer does not create it. `BBox` represents all four `BBOX_*` columns plus
`PERSON_DETECTED`.

| Dataset | SAMPLE_ID | VIDEO_ID | START | END | TEXT | SPLIT | REL_PATH | GLOSS | CLASS_ID | SIGNER_ID | FPS | BBox |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YouTube-ASL | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | — | — | — | — |
| OpenASL | ✓ | ✓ | ✓ | ✓ | ○ | ○ | — | — | — | ○ | — | ○ |
| How2Sign | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| AUTSL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| BOBSL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ○ | — | — |
| BSL-1K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | — | — |
| CSL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ | — |
| LSA64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| MS-ASL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RWTH-PHOENIX-Weather | ✓ | ✓ | ○ | ○ | ○ | ✓ | ✓ | ○ | — | ○ | ✓ | — |
| SLoVo | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ✓ | ✓ | — |
| WLASL | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

How2Sign loads a caller-supplied tab-separated file through `read_manifest()`.
It normalizes `SENTENCE_NAME`, `VIDEO_NAME`, `START_REALIGNED`,
`END_REALIGNED`, `SENTENCE`, and `CAPTION`, but neither the adapter nor the
shared reader validates the resulting columns. BSL-1K delegates to BOBSL and
forces the `isolated_signs` view.

BOBSL's `subtitle_slt` view omits `GLOSS` and `CLASS_ID`; its
`isolated_signs` view emits both. `SIGNER_ID` appears only when the isolated
annotation record provides it. `WEBVTT` is a file header recognized by the
parser, not a manifest column.

## Dataset Extensions

| Dataset | Extension columns and conditions |
|---|---|
| YouTube-ASL | None |
| OpenASL | `BBOX_X1`, `BBOX_Y1`, `BBOX_X2`, `BBOX_Y2`, `PERSON_DETECTED` when a bbox JSON exists; `SPLIT` and `SIGNER_ID` only when the source TSV contains them |
| How2Sign | All unrecognized input columns pass through unchanged |
| AUTSL | No additional columns beyond the matrix; English and Turkish class-map strings become `TEXT` and `GLOSS`, not `GLOSS_EN`/`GLOSS_TR` columns |
| BOBSL / BSL-1K | View-dependent lexical fields described above |
| CSL | `LANGUAGE` (`zh`), `VARIATION_ID` |
| LSA64 | `SOURCE_VARIANT`, `REPETITION_ID`, `HANDEDNESS` |
| MS-ASL | `SOURCE_URL`; bbox fields are present even when their values are null |
| RWTH-PHOENIX-Weather | No additional columns beyond the matrix |
| SLoVo | Optional `SRC_WIDTH`, `SRC_HEIGHT`, `FRAME_COUNT`, `FRAME_START`, `FRAME_END` copied from release annotations |
| WLASL | `SOURCE_URL`, `SOURCE`, `VARIATION_ID`, `FRAME_START`, `FRAME_END`; bbox fields are always present and may be null |

Any adapter using the shared availability helpers may also emit `AVAILABLE`
when configured with `mark_unavailable`. The default `drop_unavailable` policy
removes rows instead of adding the column.

## Known Inconsistencies

- YouTube-ASL emits neither `SPLIT` nor `REL_PATH`. This appears deliberate:
  its transcript source has no split field, and `VIDEO_ID` directly identifies
  the downloaded file. Generic video resolution therefore still works.
- WLASL emits `GLOSS` but no `TEXT`. WebDataset consequently creates an empty
  `.txt` member for every WLASL sample. The code does not state whether this is
  intentional or whether `TEXT` should mirror `GLOSS`.
- OpenASL can omit `TEXT` when the configured `text_column` is absent. It logs a
  warning and continues.
- RWTH-PHOENIX-Weather creates `START` and `END` only when its source CSV has
  valid `start` and `end` fields. The official-layout test fixture omits them,
  yet every processing stage requires them. The writer also copies those
  values without proving whether their source timebase is seconds or frames.
- How2Sign accepts any input columns and performs no post-normalization schema
  check, so a malformed external manifest can reach processing unchanged.
- `SPLIT`, gloss/class fields, signer metadata, and bbox metadata are not
  included in generic WebDataset JSON. Only identifiers, timing, processor,
  text, and processor-specific artifacts are packaged.

## Legacy Column Normalization

`read_manifest()` renames a legacy field only if the canonical name is not
already present:

| Legacy | Canonical |
|---|---|
| `SENTENCE_NAME` | `SAMPLE_ID` |
| `VIDEO_NAME` | `VIDEO_ID` |
| `START_REALIGNED` | `START` |
| `END_REALIGNED` | `END` |
| `SENTENCE` or `CAPTION` | `TEXT` |

## See Also

- [Output formats](outputs.md) -- artifacts that consume this contract
- [Pipeline stages](stages.md) -- activation and stage I/O
- [Pipeline model](../concepts/pipeline-model.md) -- context threading
