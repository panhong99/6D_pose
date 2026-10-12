#!/usr/bin/env bash
cd /home/pan/pan/6D_pose/rl
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/train.py rl \
  --bc /home/pan/pan/6D_pose/rl/logs/basket/bc_data_basket_v3_20261012/seed3_20261012_215747/best.pt \
  --starts /home/pan/pan/6D_pose/IL/basket_rl_starts_20261012 --envs 256 --eval_episodes 1024 --seed 7 \
  --eval_ckpt /home/pan/pan/6D_pose/rl/logs/basket/rl_20261013_001409/best.pt > logs/basket/rl_reeval.log 2>&1 < /dev/null
grep "re-eval" logs/basket/rl_reeval.log >> logs/basket/night_chain.log
