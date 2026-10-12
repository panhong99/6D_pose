#!/usr/bin/env bash
# v9 (Claude 2026-10-06): v8 + the arm holds an integrated TCP position target (zero action = stay; relative diff-IK sank
# ~7 mm/s under the plate).  Continues from the v8 stage-2 policy that already approaches and grasps the rim.
# Stops after stage 2 so the Human sees the GUI before stage 3.
set -euo pipefail
task_rl_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PP_VERSION=v9
export PP_BOX_H=0.1146
unset CONDA_PREFIX CONDA_DEFAULT_ENV PP_READY_START_FRAC
export TERM=xterm
exec python3 "$task_rl_dir/curriculum.py" --start_stage 2 --last_stage 2 \
  --init_checkpoint /home/pan/IsaacLab/logs/rsl_rl/m1013_2f85_rim_v8/2026-10-06_17-17-32_stage2/model_1996.pt "$@"
