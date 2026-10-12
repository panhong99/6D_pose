#!/usr/bin/env bash
set -u
cd /home/pan/pan/6D_pose/rl
printf '%s physical probe start pid=%s\n' "$(date -Is)" "$$" >> logs/basket/codex_physical_probe_detached.status
 timeout 900 env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh -u basket/expert_physical_probe_20261012.py collect --headless --speed 6 --out /home/pan/pan/6D_pose/IL/data_basket_physical_probe_detached_20261013 --batches 1 --seed 880 > logs/basket/codex_physical_probe_detached.log 2>&1
 task_probe_exit=$?
printf '%s physical probe end exit=%s\n' "$(date -Is)" "$task_probe_exit" >> logs/basket/codex_physical_probe_detached.status
exit "$task_probe_exit"
