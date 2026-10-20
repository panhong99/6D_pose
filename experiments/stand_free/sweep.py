"""(10/19 night, 보스: "흰 접시를 빨간 접시 자리로 안 옮겨도 알아서 경로") Offline map of where on the stand a plate can be
for each demo step, with the planner exactly as calibration/pipeline.py --sim_q --region (no robot, no camera).

    python experiments/stand_free/sweep.py --out DIR [--grasp_all] [--workers 6]

--grasp_all: try every valid library grasp candidate (not only the best-clearance one) until the insert plans."""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration'))
sys.path.insert(0, str(ROOT / 'rl'))

BASKET_PLAN = ROOT / 'calibration/data/rl_plans/20261019_182150_953_taught/plan.json'   # 18:21 real basket pose
STEPS = (('Q3', 'front', 'red', ()), ('Q3', 'back', 'white', ('Q3:12',)),
         ('Q4', 'back', 'red', ()), ('Q4', 'front', 'white', ('Q4:3',)))   # (10/20) red at 3


def plate_blocked(p, Tb, rim=0.134, pad=0.01):
    """Name of a cell frame box / the basket / a marker block that a flat plate (rim radius) at root p would overlap."""
    import plate_plans as pp
    from robot_model import frame_layout, measured_cell
    a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    disc = np.array([[p[0] + rr * np.cos(t), p[1] + rr * np.sin(t), p[2] + 0.01] for rr in (rim * 0.5, rim) for t in a])
    for name, c, sz, *R in frame_layout(measured_cell()) + list(pp.basket_boxes(Tb)):
        if name == 'table':
            continue
        loc = (disc - np.asarray(c)) @ (np.asarray(R[0]) if R else np.eye(3))
        if np.any(np.all(np.abs(loc) <= np.asarray(sz) / 2 + pad, axis=1)):
            return name
    return ''


def plan_step(job):
    import contextlib
    import io
    import pipeline as pl
    import plate_plans as pp
    import basket_env as be
    from calib_core import load_json, latest_t_base_cam
    (qd, region, colour, standing), (r, az_deg), grasp_all, out, extra = job
    extra = dict(extra)
    q = pl.plate_cfg('ik_bc')
    q.update({k: v for k, v in extra.items() if not k.startswith('_')})
    if colour == 'white':
        q['offset_m'] = [q['offset_m'][0] + pl.C['white_plate']['grasp_deeper_m'], q['offset_m'][1]]
    Tbc = np.array(load_json(latest_t_base_cam())['T_base_cam'])
    Tb = np.array(json.loads(BASKET_PLAN.read_text())['basket_T'])
    bdx, bdy, bdyaw = extra.pop('_basket_shift', (0.0, 0.0, 0.0))   # (10/19 night) the basket moved/turned on the table
    if bdx or bdy or bdyaw:
        c_, s_ = np.cos(np.radians(bdyaw)), np.sin(np.radians(bdyaw))
        Rz = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
        Tb = Tb.copy()
        Tb[:3, :3] = Rz @ Tb[:3, :3]
        Tb[:2, 3] += [bdx, bdy]
    p = np.array([r * np.cos(np.radians(az_deg)), r * np.sin(np.radians(az_deg)), 0.068])
    import os
    pp.DATA = Path(out) / 'scratch' / str(os.getpid())   # one folder per worker: plan dirs are named by the ms
    pp.DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    res = dict(quadrant=qd, region=region, colour=colour, r=r, az=az_deg, ok=False, why='', pin=None, grasps_tried=0,
               basket_shift=[bdx, bdy, bdyaw])
    blocked = plate_blocked(p, Tb)                          # a plate that would overlap the cell frame or the basket
    if blocked:                                             # cannot be put there at all: not a planner failure
        res.update(why=f'physical: {blocked}', t=0.0)
        return res
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lr, la = pp.library_target(p, q)
            grasps = pp.library_plans(lr, la, q, Tbc, plate_z=float(p[2])) if grasp_all else \
                [json.loads(Path(pp.plan_from_library(lr, la, q, Tbc, plate_z=float(p[2]))).read_text())]
    except Exception as e:                                       # noqa: BLE001
        res['why'] = f'grasp: {str(e)[:80]}'
        res['t'] = time.monotonic() - t0
        return res
    rq = pp.robot_quadrants(Tb)
    mq = rq[qd]
    sq = pl.C['basket_quadrant_sim'][mq]
    plates = [(rq[v.split(':')[0]], pp.pin_gap(Tb, rq[v.split(':')[0]], int(v.split(':')[1]))) for v in standing]
    same = [int(v.split(':')[1]) for v in standing if v.startswith(qd)]
    cands = pl.region_pins(len(pp.basket_gaps()[mq]), region, same, int(q.get('basket_region_min_spacing', 4)),
                           (q.get('basket_region_back_first_pin') or {}).get(qd))
    order = (-1.0, 1.0) if sq['flip'] else (1.0, -1.0)
    kinds = set()
    for legs0 in grasps:
        res['grasps_tried'] += 1
        legs0 = legs0['legs']
        H, tool = pp.hand_plate(legs0['to_grasp']['waypoints_deg'][-1], p)
        for faces_ in ((order[0],), order):
            for pin in cands:
                kw = dict(tilt_deg=sq['lean'] * be.B['full']['tine_lean_deg'][mq],
                          centre_h=sq['centre_h'] + q.get('basket_quadrant_sim_real_dh_m', 0.0),
                          x_shift_m=sq['aim_m'] + q.get('basket_quadrant_model_dx_m', {}).get(mq, 0.0), faces=faces_,
                          y_extra_m=q.get('basket_quadrant_sim_real_dy_m', {}).get(mq, 0.0))
                with contextlib.redirect_stdout(io.StringIO()):
                    e, rep = pp.quadrant_plan(Tb, mq, pp.pin_gap(Tb, mq, pin), H, tool, legs0['lift']['waypoints_deg'][-1],
                                              Tbc, q, yaw_trim_deg=0.0, y_shift_m=None,
                                              grasp_deg=legs0['to_grasp']['waypoints_deg'][-1], plates=plates,
                                              grasp_low_z=float(p[2]), **kw)
                if e is not None:
                    res.update(ok=True, pin=pin, face=rep['face'], phi=rep['phi_deg'], carry=rep['carry'][:12],
                               t=time.monotonic() - t0, plan=str(e), standing=list(standing))
                    return res
                kinds |= {w.split(':')[0] if ':' in w else w.split(' ')[0] for _, w in rep['tried']}
    res['why'] = ','.join(sorted(kinds))
    res['t'] = time.monotonic() - t0
    return res


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--out', required=True)
    a.add_argument('--grasp_all', action='store_true')
    a.add_argument('--start_lift', type=float, default=0.0, help='basket_carry_start_lift_m (opt-in)')
    a.add_argument('--basket_grid', default='', help="basket shifts 'dx,..;dy,..;dyaw,..' (m, m, deg), e.g. '-0.06,0,0.06;-0.06,0,0.06;-4,0,4'")
    a.add_argument('--extra', default='{}', help='json merged into the plate config, e.g. {"basket_rrt": {"seeds": [0, 1, 2]}}')
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--radii', default='0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25')
    a.add_argument('--steps', default='0,1,2,3', help='which of STEPS (Q3 front, Q3 back, Q4 back, Q4 front)')
    a.add_argument('--az', default='10,20,30,40')
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pos = [(float(r), float(z)) for r in a.radii.split(',') for z in a.az.split(',')]
    extra = {'basket_carry_start_lift_m': a.start_lift} if a.start_lift > 0 else {}
    extra.update(json.loads(a.extra))
    shifts = [(0.0, 0.0, 0.0)]
    if a.basket_grid:                                       # 'dx,..;dy,..;dyaw,..' (m, m, deg)
        gx, gy, gyaw = ([float(v) for v in part.split(',')] for part in a.basket_grid.split(';'))
        shifts = [(x, y, w) for x in gx for y in gy for w in gyaw]
    jobs = [(STEPS[int(i)], p_, a.grasp_all, str(out), dict(extra, _basket_shift=sh)) for i in a.steps.split(',')
            for p_ in pos for sh in shifts]
    t0 = time.time()
    with Pool(a.workers) as pool, open(out / 'results.jsonl', 'w') as f:
        for res in pool.imap_unordered(plan_step, jobs):
            f.write(json.dumps(res) + '\n')
            f.flush()
    print(f'{len(jobs)} jobs in {time.time() - t0:.0f} s -> {out}', flush=True)
