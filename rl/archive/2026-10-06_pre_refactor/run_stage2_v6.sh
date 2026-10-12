#!/usr/bin/env bash
# Start only when the Human requests training. Existing stage-1 weights are reused.
set -euo pipefail
task_rl_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PP_VERSION=v6
export PP_STAGE=2
export PP_BOX_H=0.1146
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TERM=xterm
exec python3 "$task_rl_dir/curriculum.py" --start_stage 2 \
  --init_checkpoint /home/pan/IsaacLab/logs/rsl_rl/m1013_2f85_rim_v4/2026-10-06_11-18-23_stage1/model_999.pt \
  "$@"
