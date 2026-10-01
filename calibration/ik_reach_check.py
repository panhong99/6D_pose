"""Can the arm reach the cube if the camera-based position is a few cm off?  Offline, read-only, no robot, no ROS.

    python calibration/ik_reach_check.py [--centre X Y Z] [--span 0.20] [--step 0.05] [--heights 0.10 0.15]

Takes the predicted cube centre (default: the newest dry-run plan in data/real_dryrun/), sweeps a square window of
+-span metres around it, and for every grid point solves the top-down pregrasp IK (cube yaw 0; the wrist twist j6 is the
only thing that changes with yaw) and checks
  * IK solution exists, all joints inside the URDF limits (2 deg margin),
  * clearance to the work cell (table, frame, camera body) >= margin,
  * the straight joint move from the all-zero default pose keeps >= 1.15 x margin only if --path is given.
Output: a map per pregrasp height (`.` ok, `C` too close to the cell, `L` no IK / joint limit) and a CSV with the joints.
The work cell is workcell.FRAME_CELL (measured frame and table) unless --cell_json overrides it.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

import real_pose_dryrun as dry
from ros2_pendant_mover import joint_limits_deg
from kinematics import DEFAULT_URDF, Kinematics, grasp_frames

DATA = Path(__file__).resolve().parent / 'data'


def newest_cube_centre():
    for f in sorted(glob.glob(str(DATA / 'real_dryrun' / '*' / 'plan_*.json')), reverse=True):
        plan = json.loads(Path(f).read_text())
        if plan.get('cube_centre_m'):
            return np.array(plan['cube_centre_m']), f
    raise SystemExit('no dry-run plan found: pass --centre X Y Z')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--centre', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='metres, robot base frame')
    p.add_argument('--span', type=float, default=0.20)
    p.add_argument('--step', type=float, default=0.05)
    p.add_argument('--heights', type=float, nargs='+', default=(0.10, 0.15), help='pregrasp heights above the cube centre, m')
    p.add_argument('--cell_json', type=Path, default=None)
    p.add_argument('--margin_m', type=float, default=0.10)
    p.add_argument('--tool_length_m', type=float, default=0.0)
    p.add_argument('--tool_radius_m', type=float, default=0.05)
    p.add_argument('--t_base_cam', type=Path, default=None)
    p.add_argument('--ik_only', action='store_true', help='ignore the work cell: only IK + joint limits')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    if a.centre:
        centre, src = np.array(a.centre), 'command line'
    else:
        centre, src = newest_cube_centre()
    dargs = dry.parse_args([])
    dargs.cell_json, dargs.margin_m = a.cell_json, a.margin_m
    dargs.tool_length_m, dargs.tool_radius_m = a.tool_length_m, a.tool_radius_m
    T_bc = np.array(dry.load_json(a.t_base_cam or dry.latest_t_base_cam())['T_base_cam'])
    kin = Kinematics(DEFAULT_URDF)
    checker, _ = dry.build_checker(kin, dargs, T_bc)
    lo, hi = np.array(joint_limits_deg()).T
    offsets = np.round(np.arange(-a.span, a.span + 1e-9, a.step), 4)
    out = a.out or DATA / 'real_dryrun' / f'ik_reach_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    rows = ['height_m,dx_m,dy_m,x_m,y_m,status,clearance_mm,j1,j2,j3,j4,j5,j6']
    print(f'centre {np.round(centre, 3).tolist()} m (from {src}); window +-{a.span} m, step {a.step} m; '
          f'margin {a.margin_m * 1000:.0f} mm, cell {a.cell_json or "FRAME_CELL"}\n')
    totals = {}
    for h in a.heights:
        print(f'pregrasp {h * 1000:.0f} mm above the cube centre   (columns: dx = x offset, rows: dy = y offset, +y up)')
        print('        ' + ' '.join(f'{dx * 100:+5.0f}' for dx in offsets))
        counts = dict(ok=0, C=0, L=0)
        for dy in offsets[::-1]:
            line = []
            for dx in offsets:
                c = centre + [dx, dy, 0.0]
                status, clr, q = 'L', np.nan, None
                try:
                    q = np.degrees(grasp_frames(kin, c, 0.0, a.tool_length_m, h)[3])
                    ok_limits = bool(((q > lo + 2) & (q < hi - 2)).all())
                    clr = np.inf if a.ik_only else checker.clearance(np.radians(q))
                    status = 'L' if not ok_limits else ('C' if clr < checker.margin else 'ok')
                except RuntimeError:
                    status = 'L'
                counts[status] += 1
                line.append(' . ' if status == 'ok' else f' {status} ')
                rows.append(f'{h},{dx},{dy},{c[0]:.4f},{c[1]:.4f},{status},{clr * 1000 if np.isfinite(clr) else ""},'
                            + (','.join(f'{v:.2f}' for v in q) if q is not None else ',,,,,'))
            print(f'  {dy * 100:+5.0f}  ' + ' '.join(f'{x:>5}' for x in line))
        totals[h] = counts
        print(f'  -> ok {counts["ok"]}, too close to cell (C) {counts["C"]}, no IK / joint limit (L) {counts["L"]}\n')
    out.write_text('\n'.join(rows) + '\n')
    print(f'details: {out}')


if __name__ == '__main__':
    main()
