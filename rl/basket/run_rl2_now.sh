#!/usr/bin/env bash
cd /home/pan/pan/6D_pose/rl
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/train.py rl \
  --bc /home/pan/pan/6D_pose/rl/logs/basket/bc_data_basket_v3_20261012/seed3_20261012_215747/best.pt \
  --starts /home/pan/pan/6D_pose/IL/basket_rl_starts_20261012 --envs 256 --eval_every 50 --eval_episodes 1024 --seed 3 --lr 5e-5 --patience 4 --init_ckpt /home/pan/pan/6D_pose/rl/logs/basket/rl_20261013_001409/best.pt \
  > logs/basket/night_rl2.log 2>&1 < /dev/null
echo "$(date '+%F %T') RL process exit $?" >> logs/basket/night_chain.log
grep -E "^\[rl\] (iteration 0|it [0-9]+: eval|STOP|done|dropped)" logs/basket/night_rl2.log >> logs/basket/night_chain.log
