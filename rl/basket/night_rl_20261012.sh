#!/usr/bin/env bash
# Wait for the RL start states, then residual PPO on BC v3 seed 3 (config rl, 256 envs, eval 512 episodes every 50 it).
set -u
cd /home/pan/pan/6D_pose/rl
LOG=logs/basket/night_chain.log
while pgrep -f "save_starts" > /dev/null; do sleep 30; done
echo "$(date '+%F %T') ===== RL: starts $(ls /home/pan/pan/6D_pose/IL/basket_rl_starts_20261012 | wc -l) files" >> $LOG
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/train.py rl \
  --bc /home/pan/pan/6D_pose/rl/logs/basket/bc_data_basket_v3_20261012/seed3_20261012_215747/best.pt \
  --starts /home/pan/pan/6D_pose/IL/basket_rl_starts_20261012 --envs 256 --eval_episodes 512 --eval_every 50 \
  > logs/basket/night_rl.log 2>&1 < /dev/null
grep -E "^\[rl\] (iteration 0|it [0-9]+: eval|STOP|done|dropped)" logs/basket/night_rl.log >> $LOG
echo "$(date '+%F %T') RL chain end" >> $LOG
