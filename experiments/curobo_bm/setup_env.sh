#!/bin/bash
# Isolated cuRobo env for the offline planner comparison (not the demo env).
# RTX 5070 = sm_120 -> torch cu128 + cuRobo CUDA kernels built for 12.0.
set -euo pipefail
ENV=/home/pan/pan/6D_pose/.envs/curobo-bm-20261020
SRC=/home/pan/pan/6D_pose/.envs/src/curobo
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8
~/miniconda3/bin/conda create -y -p "$ENV" python=3.10
PY=$ENV/bin/python
$PY -m pip install --upgrade pip setuptools wheel ninja
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
mkdir -p "$(dirname $SRC)"
[ -d "$SRC" ] || git clone https://github.com/NVlabs/curobo.git "$SRC"
cd "$SRC" && git log -1 --format='curobo %H %cd'
$PY -m pip install -e ".[dev,cu12]" --no-build-isolation   # cu12: cuda-core + runtime (cuRobo V2 docs)
$PY -c "import torch, curobo; print('torch', torch.__version__, torch.cuda.get_device_name(0)); print('curobo ok')"
