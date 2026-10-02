"""DRY RUN: FoundationPose pose -> robot base -> IK -> frame-collision-checked waypoint table.  Nothing moves.

    conda terminal 1:  python real_time_project/main.py --serial 338122300585 --width 1280 --height 720 ...
                       (detector + FoundationPose; sends mesh-to-camera 4x4 poses over UDP 5005)
    conda terminal 2:  python calibration/real_pose_dryrun.py [--t_base_cam .../T_base_cam.json] [--cell_json ...]
    ROS2 terminal:     python3 calibration/ros2_dryrun_publisher.py       (optional: publish the result as topics)

Chain: T_base_object = T_base_cam @ T_cam_object  ->  top-down pregrasp flange pose (cube yaw mod 90) above the
cube -> IK (URDF kinematics of the M1013) -> clearance check against the work-cell boxes (table, frame, D455
body) -> RRT path with straight joint segments -> table of joint waypoints for a HUMAN to set on the pendant.

It never sends anything to the robot.  Its only outputs are the terminal, files in data/real_dryrun/<time>/ and one
UDP JSON to 127.0.0.1:<publish_port> (read by ros2_dryrun_publisher.py, which publishes NON-control topics).
The current joint angles are only READ (ros2_flange_udp.py on UDP 5006) to start the path from where the robot is;
without them the start is all-zero joints.
"""
import argparse
import collections
import json
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import workcell as wc
from calib_utils import inv_T, load_json, rot_angle_deg, save_json
from kinematics import DEFAULT_URDF, Kinematics, cube_yaw_deg, grasp_frames
from trajectory import polyline_path

DATA = Path(__file__).resolve().parent / 'data'


def latest_t_base_cam():
    found = sorted((DATA / 'real_handeye').glob('*/T_base_cam.json'))
    if not found:
        raise SystemExit('no real hand-eye result in data/real_handeye/; pass --t_base_cam')
    # Human decision 2026-10-02: marker-scale + constant x/y offset fitted on 5 cube positions vs wrist-ArUco GT
    # (horizontal error ~7 -> ~2 mm leave-one-out; zfix was ~19 mm).  Valid for the current camera placement only.
    for name in ('T_base_cam_markerscale_xyoffset.json', 'T_base_cam_zfix.json'):
        chosen = found[-1].with_name(name)
        if chosen.is_file():
            return chosen
    return found[-1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--t_base_cam', type=Path, default=None, help='hand-eye result (default: newest real_handeye session)')
    p.add_argument('--udp_port', type=int, default=5005, help='FoundationPose poses from real_time_project/main.py')
    p.add_argument('--pose_udp_port', type=int, default=5006, help='READ-ONLY current joints from ros2_flange_udp.py')
    p.add_argument('--publish_port', type=int, default=5010, help='localhost UDP for ros2_dryrun_publisher.py')
    p.add_argument('--cell_json', type=Path, default=None,
                   help='override work-cell values (see data/real_cell_template.json); default: workcell.FRAME_CELL')
    p.add_argument('--margin_m', type=float, default=0.10, help='required keep-out distance to the cell')
    p.add_argument('--pregrasp_height_m', type=float, default=0.10)
    p.add_argument('--tool_length_m', type=float, default=0.0)
    p.add_argument('--tool_radius_m', type=float, default=0.05)
    p.add_argument('--stable_n', type=int, default=8, help='consecutive poses that must agree')
    p.add_argument('--stable_mm', type=float, default=3.0)
    p.add_argument('--stable_deg', type=float, default=3.0)
    p.add_argument('--reach_m', type=float, nargs=2, default=(0.25, 1.25), help='plausible object distance from the base')
    p.add_argument('--z_m', type=float, nargs=2, default=(-0.03, 0.15), help='plausible object centre height in base')
    p.add_argument('--max_segment_deg', type=float, default=45.0,
                   help='largest single-joint turn allowed in one point-to-point move of the pendant table')
    p.add_argument('--once', action='store_true', help='exit after the first accepted plan')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args(argv)


class UdpLatest:
    """Latest datagram of a UDP port (non-blocking helper)."""

    def __init__(self, port, timeout=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.sock.settimeout(timeout)

    def recv(self):
        try:
            return json.loads(self.sock.recv(65535))
        except (socket.timeout, ValueError):
            return None

    def drain_last(self):
        self.sock.setblocking(False)
        last = None
        try:
            while True:
                last = json.loads(self.sock.recv(65535))
        except (BlockingIOError, ValueError):
            pass
        finally:
            self.sock.settimeout(None)
        return last


def build_checker(kin, args, T_base_cam):
    cell = dict(wc.FRAME_CELL)
    if args.cell_json:
        cell.update({k: v for k, v in load_json(args.cell_json).items() if not k.startswith('_')})
    cell['collision_margin_m'] = args.margin_m
    wc.FRAME_CELL.update(cell)                      # CollisionChecker reads the module-level cell
    boxes = wc.frame_layout(cell)
    boxes.append(('d455_body', T_base_cam[:3, 3].tolist(), list(wc.D455_BOX)))   # the real camera as an obstacle
    checker = wc.CollisionChecker(kin, boxes)
    checker.set_tool(args.tool_length_m, args.tool_radius_m)
    return checker, cell


def straight_clearance(checker, waypoints):
    return float(checker.path_clearance(polyline_path(waypoints, 0.25)))


def plan_dryrun(kin, checker, T_base_obj, start_q, args):
    """Everything computed for one stable pose; returns a result dict (never raises for 'bad' poses)."""
    centre = T_base_obj[:3, 3]
    yaw, tilt = cube_yaw_deg(T_base_obj)
    res = dict(dry_run=True, T_base_object=T_base_obj.tolist(), cube_centre_m=centre.tolist(),
               cube_yaw_deg=yaw, cube_tilt_deg=tilt, start_joints_deg=np.degrees(start_q).tolist(),
               margin_m=checker.margin, pregrasp_height_m=args.pregrasp_height_m, accepted=False, reasons=[])
    dist = float(np.linalg.norm(centre[:2]))
    if not args.reach_m[0] <= dist <= args.reach_m[1]:
        res['reasons'].append(f'object {dist:.2f} m from the base is outside {args.reach_m}')
    if not args.z_m[0] <= centre[2] <= args.z_m[1]:
        res['reasons'].append(f'object height z {centre[2]:.3f} m is outside {args.z_m} (wrong hand-eye / pose?)')
    if tilt > 10:
        res['reasons'].append(f'cube tilt {tilt:.1f} deg: no flat face is up')
    if res['reasons']:
        return res
    try:
        _, target, grasp_T, goal, q_grasp = grasp_frames(kin, centre, yaw, args.tool_length_m, args.pregrasp_height_m)
    except RuntimeError as exc:
        res['reasons'].append(str(exc))
        return res
    res.update(T_base_flange_pregrasp=target.tolist(), q_pregrasp_deg=np.degrees(goal).tolist(),
               q_grasp_deg_ik_only=np.degrees(q_grasp).tolist(),
               pregrasp_clearance_m=checker.clearance(goal), start_clearance_m=checker.clearance(start_q))
    if res['pregrasp_clearance_m'] < checker.margin:
        res['reasons'].append(f'pregrasp is only {res["pregrasp_clearance_m"] * 1000:.0f} mm from the cell '
                              f'(< {checker.margin * 1000:.0f} mm): move the object or raise --pregrasp_height_m')
        return res
    if res['start_clearance_m'] < checker.margin:
        res['reasons'].append(f'start pose is only {res["start_clearance_m"] * 1000:.0f} mm from the cell')
        return res
    legs = {}
    for name, a, b in (('approach', start_q, goal), ('return', goal, start_q)):
        wp, how = checker.plan(a, b)
        wp = checker.simplify(wp, args.max_segment_deg, checker.margin * 1.15)        # few, short point-to-point moves
        legs[name] = dict(planner=how, waypoints_deg=[np.degrees(w).round(3).tolist() for w in wp],
                          min_clearance_straight_m=straight_clearance(checker, wp))
    res['legs'] = legs
    res['accepted'] = all(l['min_clearance_straight_m'] >= checker.margin for l in legs.values())
    if not res['accepted']:
        res['reasons'].append('straight-segment path comes closer than the margin to the cell')
    return res


def report(res):
    print('\n' + '=' * 78)
    c = np.array(res['cube_centre_m'])
    print(f'cube in base [m]: {np.round(c, 4).tolist()}  yaw {res["cube_yaw_deg"]:+.1f} deg  tilt {res["cube_tilt_deg"]:.1f} deg')
    if 'q_pregrasp_deg' in res:
        print(f'pregrasp flange z {res["T_base_flange_pregrasp"][2][3]:.3f} m, '
              f'clearance {res["pregrasp_clearance_m"] * 1000:.0f} mm (margin {res["margin_m"] * 1000:.0f} mm)')
        print(f'q pregrasp [deg]: {np.round(res["q_pregrasp_deg"], 1).tolist()}')
    for name, leg in res.get('legs', {}).items():
        print(f'\n{name}: {leg["planner"]}, {len(leg["waypoints_deg"])} waypoints, straight-segment min clearance '
              f'{leg["min_clearance_straight_m"] * 1000:.0f} mm')
        for k, w in enumerate(leg['waypoints_deg']):
            step = '' if k == 0 else '   turn max %5.1f deg' % np.abs(np.array(w) - np.array(leg['waypoints_deg'][k - 1])).max()
            print(f'  {k:2d}  ' + '  '.join(f'{v:8.2f}' for v in w) + step)
    print('\nRESULT:', 'ACCEPTED (dry run, nothing sent)' if res['accepted'] else 'REFUSED - ' + '; '.join(res['reasons']))
    print('=' * 78, flush=True)


def main():
    args = parse_args()
    t_path = args.t_base_cam or latest_t_base_cam()
    T_base_cam = np.array(load_json(t_path)['T_base_cam'])
    out = args.out or DATA / 'real_dryrun' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    kin = Kinematics(DEFAULT_URDF)
    checker, cell = build_checker(kin, args, T_base_cam)
    print(f'DRY RUN - nothing is sent to the robot.\nT_base_cam: {t_path}\n'
          f'cell: {args.cell_json or "workcell.FRAME_CELL (measured frame/table, assumed profile and table edge)"}  margin {args.margin_m * 1000:.0f} mm\n'
          f'listening for FoundationPose poses on UDP {args.udp_port} ...', flush=True)
    poses = UdpLatest(args.udp_port, timeout=1.0)
    joints = UdpLatest(args.pose_udp_port)
    pub = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    history, last_done, n_plans = collections.deque(maxlen=args.stable_n), None, 0
    while True:
        msg = poses.recv()
        if msg is None:
            if history:
                print('(no pose for 1 s - object lost?)', flush=True)
                history.clear()
            continue
        try:
            T_base_obj = T_base_cam @ np.array(msg['pose'])
        except (KeyError, ValueError):
            continue
        history.append(T_base_obj)
        if len(history) < args.stable_n:
            continue
        ref = history[-1]
        spread = max(np.linalg.norm(h[:3, 3] - ref[:3, 3]) * 1000 for h in history)
        spread_deg = max(rot_angle_deg(h[:3, :3].T @ ref[:3, :3]) for h in history)
        if spread > args.stable_mm or spread_deg > args.stable_deg:
            continue
        if last_done is not None and np.linalg.norm(last_done[:3, 3] - ref[:3, 3]) * 1000 < 10 \
                and rot_angle_deg(last_done[:3, :3].T @ ref[:3, :3]) < 5:
            continue                                          # same object pose as the last plan
        T_med = ref.copy()                                        # newest pose, translation = median of the stable window
        T_med[:3, 3] = np.median([h[:3, 3] for h in history], axis=0)
        cur = joints.drain_last()
        fresh = bool(cur and 'posj' in cur and time.time() - cur.get('t', 0) < 1.0)
        start_q = np.radians(cur['posj']) if fresh else np.zeros(6)
        if not fresh:
            print('NOTE: no fresh joint stream (ros2_flange_udp.py) - path starts from all-zero joints', flush=True)
        res = plan_dryrun(kin, checker, T_med, start_q, args)
        res['t_base_cam_file'] = str(t_path)
        res['cell'] = str(args.cell_json) if args.cell_json else 'FRAME_CELL'
        res['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        n_plans += 1
        report(res)
        stem = out / f'plan_{n_plans:03d}'
        save_json(stem.with_suffix('.json'), res)
        if 'legs' in res:
            with open(stem.with_suffix('.csv'), 'w') as f:
                f.write('leg,step,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg\n')
                for name, leg in res['legs'].items():
                    f.writelines(f'{name},{k},' + ','.join(str(v) for v in w) + '\n'
                                 for k, w in enumerate(leg['waypoints_deg']))
        pub.sendto(json.dumps(res).encode(), ('127.0.0.1', args.publish_port))
        last_done = T_med
        if args.once and res['accepted']:
            break


if __name__ == '__main__':
    main()
