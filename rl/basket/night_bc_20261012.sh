#!/usr/bin/env bash
# BC preflight + 3 seeds on IL/data_basket_v2_20261012, then sim evaluation of the best seed (expert.py collect --policy).
set -u
cd /home/pan/pan/6D_pose
DATA=IL/data_basket_v2_20261012
LOG=rl/logs/basket/night_chain.log
if ! conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data $DATA --preflight-only >> $LOG 2>&1; then
  echo "$(date '+%F %T') PREFLIGHT FAILED - no BC" >> $LOG; exit 1; fi
for s in 1 2 3; do
  echo "$(date '+%F %T') bc seed $s" >> $LOG
  conda run --no-capture-output -n foundationpose python rl/basket/train.py bc --data $DATA --seed $s > rl/logs/basket/night_bc_seed$s.log 2>&1
  grep "\[bc\] done" rl/logs/basket/night_bc_seed$s.log >> $LOG
done
BEST=$(python3 -c "
import json,glob
r=[(json.load(open(f))['best_val'],f) for f in glob.glob('rl/logs/basket/bc_data_basket_v2_20261012/*/summary.json')]
print(min(r)[1].replace('summary.json','best.pt'))")
echo "$(date '+%F %T') eval best $BEST (3 batches x 64)" >> $LOG
cd rl && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/expert.py collect --headless \
  --out logs/basket/bc_eval_20261012 --batches 3 --seed 77 --policy "../$BEST" > logs/basket/night_bc_eval.log 2>&1 < /dev/null
grep "^\[collect\] batch" logs/basket/night_bc_eval.log >> logs/basket/night_chain.log
echo "$(date '+%F %T') chain end" >> logs/basket/night_chain.log
