# Troubleshooting

## `No module named signdata`

Install the repository itself, not only its dependency list:

```bash
python -m pip install -e .
python -m signdata run --list-presets
```

Run commands from the repository checkout or use an absolute config path. Do
not rely on `PYTHONPATH=src`; the editable install is the supported path.

## FFmpeg is missing

`video2pose` and `video2parts` prefer FFmpeg decoding but fall back to OpenCV.
`video2crop` and `video2compression` invoke the FFmpeg executable and cannot
produce their outputs without it.

```bash
ffmpeg -version
ffprobe -version
```

Install both through the operating system package manager and ensure they are
on `PATH`.

## YouTube transcripts are blocked

`youtube-transcript-api` may raise `RequestBlocked` or `IpBlocked`. Configure a
proxy in the job:

```yaml
dataset:
  source:
    transcript_proxy_http: http://user:pass@host:port
    transcript_proxy_https: http://user:pass@host:port
    stop_on_transcript_block: true
```

Use a permitted, non-blocked network endpoint. With
`stop_on_transcript_block: true`, acquisition stops at the first block rather
than producing a misleadingly partial corpus.

## CUDA, MMPose, or MMDet import failure

The base install does not include OpenMMLab packages. Use the repository's
supported stack from the [installation guide](installation.md): MMCV 2.0.1,
MMEngine 0.10.7, MMDetection 3.1.0, and MMPose 1.3.2. Confirm that the PyTorch
build, CUDA runtime, and GPU driver are mutually compatible before debugging a
job config.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from mmdet.apis import init_detector; print('MMDet OK')"
python -c "from mmpose.apis import init_model; print('MMPose OK')"
```

## AV1 video is black in VLC

On NVIDIA/VDPAU systems, VLC can select a hardware decoding path with no AV1
profile. The file is usually not corrupt. Disable hardware decoding for the
check:

```bash
vlc --avcodec-hw=none path/to/video.mp4
```

The equivalent UI setting is **Tools → Preferences → Input/Codecs →
Hardware-accelerated decoding → Disable**.

## `preset` is rejected while loading the config

Preset syntax depends on `processing.video_config.codec` and is validated
before work starts:

| Codec family | Accepted form | Direction |
|---|---|---|
| `*_nvenc` | `p1` through `p7` | `p1` fastest, `p7` best quality |
| `libsvtav1` | integer `-2` through `13` | `13` fastest |
| `libaom-av1` | integer `0` through `8` | `8` fastest |
| `libvpx-vp9` | integer `-8` through `8` | `8` fastest |
| Other software encoders | FFmpeg preset name such as `medium` | Encoder-specific |

See [configuration reference](../reference/configuration.md#processingvideo_config)
for the complete table and quality controls.

## A rerun skips existing samples

Processors and normalization resume at the artifact level. Pass `--force-all`
to regenerate existing outputs:

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml --force-all
```

See [Pipeline Model](../concepts/pipeline-model.md#resume-and-force-behavior) for
which artifact set counts as complete for each processor.

## A manifest loads but processing fails on timestamps

Generic consumers require canonical `START` and `END` columns in seconds.
Legacy How2Sign names are normalized, but arbitrary aliases and frame indices
are not. Inspect the [manifest contract](../reference/manifest.md) before
reusing an external manifest.
