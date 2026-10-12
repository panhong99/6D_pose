#!/usr/bin/env bash
# Overnight 10/12 -> 10/13: two demo collectors in parallel (clean + DART), then BC preflight + 3 seeds.
set -u
cd /home/pan/pan/6D_pose/rl
OUT=/home/pan/pan/6D_pose/IL/data_basket_v2_20261012
LOG=logs/basket/night_chain.log
ISAAC="/home/pan/IsaacLab/_isaac_sim/python.sh"
mkdir -p "$OUT" logs/basket
echo "$(date '+%F %T') start: clean seed 10 + dart seed 20, 30 batches x 64 envs each" >> $LOG
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm $ISAAC basket/expert.py collect --headless --out "$OUT" --batches 30 --seed 10 > logs/basket/night_collect_clean.log 2>&1 &
P1=$!
sleep 60
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm $ISAAC basket/expert.py collect --headless --out "$OUT" --batches 30 --seed 20 --dart > logs/basket/night_collect_dart.log 2>&1 &
P2=$!
echo "$(date '+%F %T') collectors pid clean $P1 dart $P2" >> $LOG
wait $P1; echo "$(date '+%F %T') clean done (exit $?)" >> $LOG
wait $P2; echo "$(date '+%F %T') dart done (exit $?)" >> $LOG
grep -h "^\[collect\] batch" logs/basket/night_collect_clean.log logs/basket/night_collect_dart.log | awk '{p+=$7; s+=$9} END {print "collect total planned", p, "inserted", s}' >> $LOG
cd /home/pan/pan/6D_pose
if conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data "$OUT" --preflight-only >> rl/$LOG 2>&1; then
  for s in 1 2 3; do
    echo "$(date '+%F %T') bc seed $s" >> rl/$LOG
    conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data "$OUT" --seed $s >> rl/logs/basket/night_bc_seed$s.log 2>&1
    tail -1 rl/logs/basket/night_bc_seed$s.log >> rl/$LOG
  done
else
  echo "$(date '+%F %T') PREFLIGHT FAILED - no BC training" >> rl/$LOG
fi
echo "$(date '+%F %T') chain end" >> rl/$LOG
