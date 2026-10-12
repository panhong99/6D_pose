#!/usr/bin/env bash
# v8 (Human decision 2026-10-06, option A): home -> ready is joint interpolation, RL learns stage 2 from the ready joints
# (training, gate evaluation and GUI all start there).  Stops after stage 2 so the Human sees the GUI before stage 3.
set -euo pipefail
task_rl_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PP_VERSION=v8
export PP_BOX_H=0.1146
unset CONDA_PREFIX CONDA_DEFAULT_ENV PP_READY_START_FRAC
export TERM=xterm
# Random init: the v7 stage-1 weights swing the arm ~15 cm off the ready pose in the first 0.5 s (their episode-start
# habit from home), then settle 100 mm short of the rim (Claude trace 2026-10-06 15:35).
exec python3 "$task_rl_dir/curriculum.py" --start_stage 2 --last_stage 2 --scratch "$@"
