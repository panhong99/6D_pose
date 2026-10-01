"""Print the robot-base coordinates PREDICTED from FoundationPose, to compare with a pose you set by hand.  Read-only.

    python calibration/predict_base_pose.py                       # live prediction (needs real_time_project/main.py)
    python calibration/predict_base_pose.py --compare "x y z a b c"   # + difference to the flange posx you read on the pendant
    python calibration/predict_base_pose.py --compare "..." --offset 0 0 50   # flange is 50 mm above the cube centre

Chain: T_base_object = T_base_cam @ T_cam_object.  T_cam_object arrives over UDP 5005 from real_time_project/main.py,
T_base_cam is the real hand-eye result (newest data/real_handeye/*/T_base_cam.json unless --t_base_cam).
Stop real_pose_dryrun.py first: only one program can read UDP 5005.  Nothing is sent to the robot or to ROS.
Units: metres in the files, millimetres and degrees on screen (Doosan posx: x y z [mm], a b c = intrinsic ZYZ [deg]).
"""
import argparse
import collections
import json
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calib_utils import T_to_posx, load_json, posx_to_T, rot_angle_deg
from real_pose_dryrun import latest_t_base_cam
from kinematics import cube_yaw_deg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--t_base_cam', type=Path, default=None)
    p.add_argument('--udp_port', type=int, default=5005)
    p.add_argument('--n', type=int, default=30, help='poses averaged per report')
    p.add_argument('--compare', default=None, help='"x y z a b c" flange posx from the pendant (mm, deg, ZYZ)')
    p.add_argument('--offset', type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=('DX', 'DY', 'DZ'),
                   help='mm, base axes: where the flange is relative to the cube centre (default: at the centre)')
    p.add_argument('--once', action='store_true')
    return p.parse_args(argv)


def collect(sock, n, timeout=10.0):
    poses, end = [], time.time() + timeout
    while len(poses) < n and time.time() < end:
        try:
            poses.append(np.array(json.loads(sock.recv(65535))['pose']))
        except (socket.timeout, ValueError, KeyError):
            continue
    return poses


def main():
    args = parse_args()
    t_path = args.t_base_cam or latest_t_base_cam()
    T_bc = np.array(load_json(t_path)['T_base_cam'])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', args.udp_port))
    sock.settimeout(1.0)
    print(f'T_base_cam: {t_path}\nwaiting for FoundationPose poses on UDP {args.udp_port} (main.py running?) ...', flush=True)
    while True:
        poses = collect(sock, args.n)
        if len(poses) < args.n // 2:
            print('not enough poses - is real_time_project/main.py tracking the cube?', flush=True)
            continue
        T_bo = [T_bc @ p for p in poses]
        t = np.array([T[:3, 3] for T in T_bo])
        spread_mm = t.std(axis=0) * 1000
        T = T_bo[-1].copy()
        T[:3, 3] = np.median(t, axis=0)
        yaw, tilt = cube_yaw_deg(T)
        cam_obj = np.median([p[:3, 3] for p in poses], axis=0)
        print('\n' + '-' * 70)
        print(f'{len(poses)} poses averaged | camera->cube {np.round(cam_obj * 1000, 1).tolist()} mm '
              f'(distance {np.linalg.norm(cam_obj) * 1000:.0f} mm)')
        print(f'PREDICTED cube centre in robot base [mm]: x {T[0, 3] * 1000:8.1f}  y {T[1, 3] * 1000:8.1f}  '
              f'z {T[2, 3] * 1000:8.1f}      (jitter std {np.round(spread_mm, 2).tolist()} mm)')
        print(f'cube yaw about base z {yaw:+.1f} deg (mod 90), tilt {tilt:.1f} deg; as posx ZYZ [deg]: '
              f'{np.round(T_to_posx(T)[3:], 1).tolist()}')
        if args.compare:
            vals = [float(v) for v in args.compare.replace(',', ' ').split()]
            if len(vals) != 6:
                raise SystemExit('--compare needs 6 numbers: x y z a b c')
            flange = posx_to_T(vals)
            expected = T[:3, 3] + np.array(args.offset) / 1000.0
            d = (expected - flange[:3, 3]) * 1000
            print(f'\nYOU SET flange posx: {np.round(vals, 2).tolist()}  (offset to cube centre {list(args.offset)} mm)')
            print(f'predicted flange position [mm]: {np.round(expected * 1000, 1).tolist()}')
            print(f'prediction - yours [mm]:  dx {d[0]:+7.1f}  dy {d[1]:+7.1f}  dz {d[2]:+7.1f}   '
                  f'(norm {np.linalg.norm(d):.1f} mm, horizontal {np.linalg.norm(d[:2]):.1f} mm)')
            print('  dz / dx / dy > 0 means FoundationPose + calibration put the cube further +z / +x / +y than you did.')
        print('-' * 70, flush=True)
        if args.once:
            break


if __name__ == '__main__':
    main()
