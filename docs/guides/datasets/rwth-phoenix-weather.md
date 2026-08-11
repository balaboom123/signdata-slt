# RWTH-PHOENIX-Weather

Continuous German Sign Language (DGS) weather-report dataset with ~8,000 sentence-level clips across train / dev / test splits ([Camgoz et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html)).

**Default pose job:** `dataset.download (materialize missing videos) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/rwth_phoenix_weather/mediapipe.yaml \
  --override dataset.source.release_dir=/path/to/PHOENIX-2014-T-release3
```

**Default video job:** `dataset.download (materialize missing videos) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/rwth_phoenix_weather/video.yaml \
  --override dataset.source.release_dir=/path/to/PHOENIX-2014-T-release3
```

**Setup:**
1. Download the public 39 GB `phoenix-2014-T.v3.tar.gz` archive from [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). Allow space for both the archive and its unpacked contents.
2. Unpack the archive. The official release stores annotations under `annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv` and frames under `features/fullFrame-210x260px/<split>/<clip>/`. The much smaller `phoenix-2014.v3-nofeatures.tar.gz` download is a different, non-`T` release and cannot satisfy this adapter.
3. Set `dataset.source.release_dir` to the unpacked release root, either via CLI override or in YAML.
4. Set `paths.videos` to the directory where materialized `.mp4` files should be written.

Repackaged layouts with top-level corpus CSVs and simpler relative frame paths are also accepted for compatibility, but the official release layout is the primary target.

The shipped RWTH-PHOENIX-Weather configs default to `dataset.source.prepare_mode: materialize_missing` so a fresh unpacked release produces `.mp4` clips before manifest filtering.

The adapter supports three prepare modes via `dataset.source.prepare_mode`:

- `validate` checks that `release_dir` exists and skips frame materialization.
- `materialize_missing` (default) encodes frame directories into `.mp4` files for clips that do not yet have a video file.
- `rematerialize_all` force re-encodes all clips regardless of existing files.

Frame directories are read in lexicographic order of their `.png` filenames. The default frame rate is 25 fps via `dataset.source.video_fps`.

