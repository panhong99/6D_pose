"""Tesollo hand, plate: from the lifted grasp to the basket, down to the pins - the 2F-85 basket planner
(calibration/plate_plans.quadrant_plan) with the Tesollo grasp instead of the gripper's:
  - grasp joints = the taught plate grasp (tesollo/arm_poses.yaml), moved by --dx/--dy like arm_plan.py
  - H (plate in the hand) from those joints and the plate centre (flat plate, x toward the hand), the same rule as
    plate_plans.hand_plate, with the hand's own length / radius for the collision model
  - basket pose = a basket_vision.json (default: the 11/3 fixed basket; a live run's measurement with --basket)
Legs (one plan.json, dry run): to_above (carry), descend (straight down, plate upright), ascend, to_home, abort_home.
Read-only (no robot command).
    ~/miniconda3/envs/foundationpose/bin/python tesollo/basket_plan.py --quadrant Q3 --pin 6 [--dx M --dy M]
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'calibration'))
os.chdir(HERE.parent / 'calibration')   # calib_core / plate_plans read their files relative to calibration/
import numpy as np
import yaml
import calib_core as cc
import plate_plans as pp

# DG-5F-S palm as a cylinder on the flange for the collision model (2F-85: 0.165 / 0.06).  At the taught plate grasp the
# hand axis is horizontal and the plate centre sits 16 cm along it, 8 cm off it: a 20 cm cylinder runs through the held
# plate itself (every insert refused, 10/27), so only the palm (12 cm) is modelled; the fingers hold the top rim.
HAND_LENGTH_M, HAND_RADIUS_M = 0.12, 0.06


def shifted_grasp(k, grasp, dx, dy):
    """The taught grasp joints moved by (dx, dy) in base, same hand orientation."""
    if not (dx or dy):
        return np.asarray(grasp, float)
    T = k.fk(np.radians(grasp))
    T[0, 3] += dx
    T[1, 3] += dy
    q = k.ik(T, np.radians(grasp))
    assert q is not None and np.abs(k.fk(q)[:3, 3] - T[:3, 3]).max() < 1e-3, 'IK failed for the shifted grasp'
    return np.degrees(q)


def lift_line(k, q0, dz, n=11):
    """Straight up from the FK of q0 (as arm_plan.py)."""
    T0, out, q = k.fk(np.radians(q0)), [np.asarray(q0, float)], np.radians(q0)
    for s in np.linspace(0, 1, n)[1:]:
        T = T0.copy()
        T[2, 3] += s * dz
        q = k.ik(T, q)
        assert q is not None, 'IK failed on the lift'
        out.append(np.degrees(q))
    return out


def hand_plate(k, grasp_deg, centre, tool):
    """(H) plate frame (centre, z up, x toward the hand) in the tool frame at the grasp - plate_plans.hand_plate's rule."""
    T_tcp = k.fk(np.radians(grasp_deg)) @ tool
    n = np.array([0.0, 0.0, 1.0])
    c = np.asarray(centre, float)
    a = T_tcp[:3, 3] - c
    a -= (a @ n) * n
    x = a / np.linalg.norm(a)
    T_pl = np.eye(4)
    T_pl[:3, :3] = np.stack([x, np.cross(n, x), n], 1)
    T_pl[:3, 3] = c
    return np.linalg.inv(T_tcp) @ T_pl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quadrant', default='Q3', choices=('Q1', 'Q2', 'Q3', 'Q4'))
    p.add_argument('--pin', type=int, default=6, help='pin pair (1 = farthest from the robot), as the demo')
    p.add_argument('--dx', type=float, default=0.0)
    p.add_argument('--dy', type=float, default=0.0)
    p.add_argument('--plate', type=float, nargs=3, help='plate centre (base, m) at the grasp; default: saved reference + dx/dy')
    p.add_argument('--rise', type=float, default=0.12, help='lift before the carry (arm_plan.py lift)')
    p.add_argument('--basket', default=str(HERE.parent / 'calibration/data/basket/demo_basket_fixed.json'))
    a = p.parse_args()
    A = yaml.safe_load((HERE / 'arm_poses.yaml').read_text())['plate']
    if a.plate is None:
        if 'obj_centre' not in A:
            sys.exit('no plate reference: tesollo/perceive.py --object plate --save_ref first, or give --plate x y z')
        a.plate = [A['obj_centre'][0] + a.dx, A['obj_centre'][1] + a.dy, A['obj_centre'][2]]
    k = pp.Kinematics()
    grasp = shifted_grasp(k, A['posj_deg'], a.dx, a.dy)
    lift = lift_line(k, grasp, a.rise)
    home = list(map(float, pp.C['robot']['home_joints_deg']))
    tool = np.eye(4)
    tool[2, 3] = HAND_LENGTH_M
    centre = np.array(a.plate) - [0.0, 0.0, 0.015]            # perceive gives the rim top: the plate middle ~1.5 cm lower
    H = hand_plate(k, grasp, centre, tool)
    q = dict(pp.C['plate_ik_bc'], tool_length_m=HAND_LENGTH_M, tool_radius_m=HAND_RADIUS_M)
    Tbb = np.array(json.loads(Path(a.basket).read_text())['T'])
    model_qd = pp.robot_quadrants(Tbb)[a.quadrant] if hasattr(pp, 'robot_quadrants') else a.quadrant
    gap = pp.pin_gap(Tbb, model_qd, a.pin)
    Tbc = np.array(cc.load_json(cc.latest_t_base_cam())['T_base_cam'])
    print(f'basket {Tbb[:3, 3].round(3).tolist()} from {a.basket}; {a.quadrant} (model {model_qd}) pin {a.pin} -> gap {gap}; '
          f'plate centre {np.round(centre, 3).tolist()}', flush=True)
    grasp_legs = {'to_grasp': [home, list(map(float, grasp))], 'lift': [list(map(float, w)) for w in lift]}
    path, rep = pp.quadrant_plan(Tbb, model_qd, gap, H, tool, list(map(float, lift[-1])), Tbc, q, grasp_legs=grasp_legs,
                                 grasp_deg=list(map(float, grasp)))
    if path is None:
        sys.exit(f'NO PLAN: ' + str(__import__('collections').Counter(t[1].split(' ')[0] for t in rep.get('tried', []))) + f' over {len(rep.get("tried", []))} tries')
    plan = json.loads(Path(path).read_text())
    print(f'OK: phi {rep.get("phi_deg")}, carry {rep.get("carry")}, accepted {rep.get("accepted")}, '
          f'clearance/margin {rep.get("clearance_ratio")}')
    for leg, v in plan['legs'].items():
        w = v['waypoints_deg']
        print(f'  {leg}: {len(w)} waypoints, end {[round(x, 1) for x in w[-1]]}')
    print('plan:', path)


if __name__ == '__main__':
    main()
