#!/usr/bin/env bash
cd /home/pan/pan/6D_pose/rl
printf '%s start pid=%s\n' "$(date -Is)" "$$" >> logs/basket/codex_physical_verified.status
timeout 900 env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh -u basket/expert_physical_verified_20261013.py collect --headless --speed 2.5 --episode_s 90 --out /home/pan/pan/6D_pose/IL/data_basket_physical_verified_20261013 --batches 2 --seed 880 > logs/basket/codex_physical_verified.log 2>&1
task_trace_exit=$?
printf '%s end exit=%s\n' "$(date -Is)" "$task_trace_exit" >> logs/basket/codex_physical_verified.status

