# Installation Guide

## Base Installation

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs all dependencies needed for MediaPipe-based extraction, which works on CPU out of the box.

## MMPose (GPU Required)

MMPose RTMPose3D requires a CUDA-capable GPU. Follow these steps after the base installation:

### 1. Install MMPose dependencies

```bash
pip install -U openmim
mim install mmcv==2.0.1 mmengine==0.10.7 mmdet==3.1.0
```

### 2. Clone and install MMPose

```bash
git clone https://github.com/open-mmlab/mmpose.git ../mmpose
pip install -v -e ../mmpose
export PYTHONPATH="/path/to/mmpose:$PYTHONPATH"
```

### 3. Download model checkpoints

```bash
wget -P src/signdata/models/checkpoints/ \
  https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth

wget -P src/signdata/models/checkpoints/ \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
```

These checkpoints are used by the RTMPose3D whole-body estimator and the RTMDet person detector respectively.

### 4. Verify installation

```bash
python -c "from mmpose.apis import init_model; print('MMPose OK')"
python -c "from mmdet.apis import init_detector; print('MMDet OK')"
```

Both commands should print without errors. If you see CUDA-related issues, verify your GPU driver and PyTorch CUDA version match.

---

## See Also

- [Architecture](architecture.md) -- system design and pipeline flow
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Datasets](datasets.md) -- dataset-specific setup guides
