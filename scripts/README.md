# Repository Scripts

These are maintainer and research utilities, not installed `signdata` entry
points. Run them from the repository root in an editable development install.

| Script | Purpose | Portability |
|---|---|---|
| `gen_docs_reference.py` | Regenerate schema, CLI, and preset reference blocks | Maintained; no dataset, network, or GPU required |
| `benchmark_nvenc_profiles.py` | Benchmark codec profiles and emit compression reports | Research utility; requires FFmpeg, NVIDIA hardware, and explicit local paths |
| `evaluate_mediapipe_compression.py` | Compare MediaPipe landmarks across compression conditions | Research utility; requires a prepared local corpus and explicit local paths |
| `download_with_new_rule.py` | Audit videos against an older yt-dlp selection experiment | Legacy utility; contains workstation defaults and imports the untracked `src/asl_prep` package, so it is not runnable from a clean clone |

The compression and landmark-evaluation scripts produced the dated
[YouTube-ASL corpus audit](../docs/notes/reports/youtube-asl-corpus-audit.md).
Their defaults record that experiment's workstation layout; supply command-line
paths instead of treating those defaults as repository configuration.

Regenerate code-derived documentation with:

```bash
python scripts/gen_docs_reference.py
```
