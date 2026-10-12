#!/usr/bin/env bash
# v7 (Claude 2026-10-06): retrain stage 1 at the 114.6 mm support (stage-1 model_999 from 54.6 mm scores 0 % there),
# then stage 2 with half of the TRAINING episodes starting at the ready pose (gate eval and video start at home).
set -euo pipefail
task_rl_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PP_VERSION=v7
export PP_BOX_H=0.1146
export PP_READY_START_FRAC=0.5
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TERM=xterm
exec python3 "$task_rl_dir/curriculum.py" --start_stage 1 --last_stage 2 \
  --init_checkpoint /home/pan/IsaacLab/logs/rsl_rl/m1013_2f85_rim_v4/2026-10-06_11-18-23_stage1/model_999.pt \
  "$@"
