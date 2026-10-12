#!/usr/bin/env bash
cd /home/pan/pan/6D_pose/rl
printf '%s start pid=%s\n' "$(date -Is)" "$$" >> logs/basket/codex_physical_slowstart.status
timeout 900 env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh -u basket/expert_physical_slowstart_20261013.py collect --headless --speed 6 --episode_s 90 --out /home/pan/pan/6D_pose/IL/data_basket_physical_slowstart_20261013 --batches 1 --seed 880 > logs/basket/codex_physical_slowstart.log 2>&1
task_trace_exit=$?
printf '%s end exit=%s\n' "$(date -Is)" "$task_trace_exit" >> logs/basket/codex_physical_slowstart.status

# Independent surrogate checkpoint evaluation after this physical probe finishes.
timeout 1200 env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh -u basket/train.py rl --bc /home/pan/pan/6D_pose/rl/logs/basket/bc_data_basket_v3_20261012/seed3_20261012_215747/best.pt --starts /home/pan/pan/6D_pose/IL/basket_rl_starts_20261012 --envs 128 --eval_episodes 1024 --seed 11 --eval_ckpt /home/pan/pan/6D_pose/rl/logs/basket/rl_20261013_003636/best.pt > logs/basket/codex_surrogate_reeval_128.log 2>&1
printf "%s reeval exit=%s\n" "$(date -Is)" "$?" >> logs/basket/codex_physical_slowstart.status
