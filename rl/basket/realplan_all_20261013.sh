#!/usr/bin/env bash
# realplan (learned policy) on the measured conditions of every labelled real run of 10/12 (sim only)
cd /home/pan/pan/6D_pose
OUTD=rl/logs/basket/realplan_all; mkdir -p $OUTD
python3 - <<'PY'
import json, os
for l in open('calibration/data/real_pipeline/labels_20261012.jsonl'):
    r = json.loads(l); d = 'calibration/data/real_pipeline/' + r['run']
    ev = [json.loads(x) for x in open(d + '/robot_state.jsonl') if '"event"' in x]
    pe = next(e for e in ev if e['event'] == 'plan'); sp = next(e for e in ev if e['event'] == 'slot_plan')
    g = json.load(open(pe['plan']))['legs']; s = json.load(open(sp['plan']))['legs']
    ph = d + '/wrist_above.json' if os.path.exists(d + '/wrist_above.json') else d + '/wrist_pre.json'
    req = dict(T_base_basket=json.load(open(ph))['T_base_basket'], grasp_deg=g['to_grasp']['waypoints_deg'][-1],
               plate_root_fp=pe['plate_root_fp'], above_deg=s['to_above']['waypoints_deg'][-1], slot_shift_pitches=2,
               out=f'/home/pan/pan/6D_pose/rl/logs/basket/realplan_all/{r["run"]}.json', real=r['outcome'])
    json.dump(req, open(f'rl/logs/basket/realplan_all/{r["run"]}.req.json', 'w'))
PY
cd rl
for f in logs/basket/realplan_all/*.req.json; do
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm /home/pan/IsaacLab/_isaac_sim/python.sh basket/expert.py realplan --headless --per_run 8 \
    --req $f --policy logs/basket/bc_data_basket_v3_20261012/seed3_20261012_215747/best.pt --residual logs/basket/rl_20261013_003636/best.pt \
    > ${f%.req.json}.log 2>&1 < /dev/null
  echo "$(basename $f .req.json) $(grep '^\[realplan\] exact' ${f%.req.json}.log | cut -c1-120)" >> logs/basket/realplan_all/summary.txt
done
echo done >> logs/basket/realplan_all/summary.txt
