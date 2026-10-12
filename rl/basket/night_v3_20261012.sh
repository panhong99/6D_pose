#!/usr/bin/env bash
# Overnight 10/12 -> 10/13: two demo collectors in parallel (clean + DART), then BC preflight + 3 seeds.
set -u
cd /home/pan/pan/6D_pose/rl
OUT=/home/pan/pan/6D_pose/IL/data_basket_v3_20261012
LOG=logs/basket/night_chain.log
ISAAC="/home/pan/IsaacLab/_isaac_sim/python.sh"
mkdir -p "$OUT" logs/basket
echo "$(date '+%F %T') start: clean seed 10 + dart seed 20, 30 batches x 64 envs each" >> $LOG
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm $ISAAC basket/expert.py collect --headless --out "$OUT" --batches 30 --seed 30 > logs/basket/v3_collect_clean.log 2>&1 < /dev/null &
P1=$!
sleep 60
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm $ISAAC basket/expert.py collect --headless --out "$OUT" --batches 30 --seed 40 --dart > logs/basket/v3_collect_dart.log 2>&1 < /dev/null &
P2=$!
echo "$(date '+%F %T') collectors pid clean $P1 dart $P2" >> $LOG
wait $P1; echo "$(date '+%F %T') clean done (exit $?)" >> $LOG
wait $P2; echo "$(date '+%F %T') dart done (exit $?)" >> $LOG
grep -h "^\[collect\] batch" logs/basket/v3_collect_clean.log logs/basket/v3_collect_dart.log | awk '{p+=$7; s+=$9} END {print "collect total planned", p, "inserted", s}' >> $LOG
cd /home/pan/pan/6D_pose
if conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data "$OUT" --preflight-only >> rl/$LOG 2>&1; then
  for s in 1 2 3; do
    echo "$(date '+%F %T') bc seed $s" >> rl/$LOG
    conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data "$OUT" --seed $s >> rl/logs/basket/v3_bc_seed$s.log 2>&1
    tail -1 rl/logs/basket/v3_bc_seed$s.log >> rl/$LOG
  done
else
  echo "$(date '+%F %T') PREFLIGHT FAILED - no BC training" >> rl/$LOG
fi
BEST=$(python3 -c "
import json,glob
r=[(json.load(open(f))['best_val'],f) for f in glob.glob('rl/logs/basket/bc_data_basket_v3_20261012/*/summary.json')]
print(min(r)[1].replace('summary.json','best.pt'))")
echo "$(date '+%F %T') eval $BEST" >> rl/$LOG
cd rl && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/expert.py collect --headless --out logs/basket/v3_bc_eval --batches 3 --seed 77 --policy "../$BEST" > logs/basket/v3_bc_eval.log 2>&1 < /dev/null
grep "^\[collect\] batch" logs/basket/v3_bc_eval.log >> $LOG
cd ..
echo "$(date '+%F %T') chain end" >> rl/$LOG
