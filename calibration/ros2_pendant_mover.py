"""SUPERVISED point-to-point mover for the REAL M1013 (system Python + ROS2).  Moves ONLY when a human types `go`.

Executes the joint waypoint table of one leg of a dry-run plan (calibration/real_pose_dryrun.py) as a series of
single MoveJoint calls, SLOWEST speed, one waypoint per confirmation.  Default is a CHECK ONLY (nothing is sent):

  python3 calibration/ros2_pendant_mover.py --plan calibration/data/real_dryrun/<time>/plan_001.json --leg approach
  python3 calibration/ros2_pendant_mover.py --plan ... --leg approach --execute        # real motion, per-step `go`

Safety design (every item can only make it refuse / stop, never move more):
  * --execute is required; without it the script only validates and prints the table (works without ROS).
  * Speed is capped: --vel_deg_s default/limit 2 deg/s; --acc_deg_s2 default/limit 2. Higher is refused.
  * The plan must be `accepted`, younger than --max_plan_age_min, and every move must turn a joint <= --max_step_deg.
  * Every waypoint must be inside the URDF joint limits (with a 2 deg margin).
  * The robot's CURRENT joints (read-only service) must match the leg's first waypoint within --max_start_error_deg,
    otherwise it refuses (the plan was made for another start pose).  It is re-checked before every single move.
  * One typed `go` per waypoint (anything else aborts); an opening statement must be typed once.  Ctrl+C aborts the
    sequence and requests a stop; the physical E-stop / pendant always has priority.
  * After each move the reached joints must match the target within --max_reach_error_deg or the sequence stops.
  * If the controller refuses (e.g. access control belongs to the pendant) the sequence stops - it never retries.
This script is only ever started by the human.  The assistant does not run it against the robot.
"""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2/urdf/m1013.urdf')
VEL_HARD_LIMIT, ACC_HARD_LIMIT = 2.0, 2.0
OPENING = 'area clear, e-stop in hand'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--leg', choices=('approach', 'return'), required=True)
    p.add_argument('--execute', action='store_true', help='really send the moves (otherwise check only)')
    p.add_argument('--vel_deg_s', type=float, default=2.0)
    p.add_argument('--acc_deg_s2', type=float, default=2.0)
    p.add_argument('--max_step_deg', type=float, default=45.0)
    p.add_argument('--max_start_error_deg', type=float, default=3.0)
    p.add_argument('--max_reach_error_deg', type=float, default=1.0)
    p.add_argument('--max_plan_age_min', type=float, default=30.0)
    p.add_argument('--from_step', type=int, default=1, help='first waypoint index to move to (1 = first move)')
    p.add_argument('--robot_id', default='dsr01')
    p.add_argument('--controller_ns', default='dsr_controller2')
    return p.parse_args(argv)


def joint_limits_deg():
    limits = {}
    for joint in ET.parse(URDF).getroot().iter('joint'):
        lim = joint.find('limit')
        if joint.get('type') == 'revolute' and lim is not None and joint.get('name', '').startswith('joint_'):
            limits[joint.get('name')] = (np.degrees(float(lim.get('lower'))), np.degrees(float(lim.get('upper'))))
    return [limits[f'joint_{i}'] for i in range(1, 7)]


def validate(args, plan):
    """Return (waypoints Nx6 deg, list of problems).  Never raises on a bad plan."""
    problems = []
    if not np.isfinite(args.vel_deg_s) or args.vel_deg_s <= 0 or args.vel_deg_s > VEL_HARD_LIMIT:
        problems.append(f'--vel_deg_s {args.vel_deg_s} outside (0, {VEL_HARD_LIMIT}] (hard limit)')
    if not np.isfinite(args.acc_deg_s2) or args.acc_deg_s2 <= 0 or args.acc_deg_s2 > ACC_HARD_LIMIT:
        problems.append(f'--acc_deg_s2 {args.acc_deg_s2} outside (0, {ACC_HARD_LIMIT}] (hard limit)')
    if not plan.get('accepted'):
        problems.append('plan is not accepted: ' + '; '.join(plan.get('reasons', [])))
    if not plan.get('dry_run'):
        problems.append('file is not a dry-run plan')
    if plan.get('offline_test') or plan.get('execution_allowed') is False:
        problems.append('offline fixture cannot be used for real motion')
    try:
        created = time.mktime(time.strptime(plan['created'], '%Y-%m-%d %H:%M:%S'))
        age = (time.time() - created) / 60
        if age < -1 or age > args.max_plan_age_min:
            problems.append(f'plan is {age:.0f} min old (> {args.max_plan_age_min:.0f}): scene may have changed, re-plan')
    except (KeyError, ValueError):
        problems.append('plan has no creation time')
    leg = plan.get('legs', {}).get(args.leg)
    if not leg:
        return np.zeros((0, 6)), problems + [f'plan has no leg {args.leg}']
    wp = np.array(leg['waypoints_deg'], dtype=float)
    if wp.ndim != 2 or wp.shape[1] != 6 or len(wp) < 2 or not np.isfinite(wp).all():
        return np.zeros((0, 6)), problems + ['waypoint table is malformed']
    lo, hi = np.array(joint_limits_deg()).T
    for k, w in enumerate(wp):
        if (w < lo + 2).any() or (w > hi - 2).any():
            problems.append(f'waypoint {k} is within 2 deg of / beyond a joint limit: {np.round(w, 1).tolist()}')
    for k in range(1, len(wp)):
        turn = np.abs(wp[k] - wp[k - 1]).max()
        if turn > args.max_step_deg:
            problems.append(f'move {k} turns a joint {turn:.1f} deg (> {args.max_step_deg})')
    if leg['min_clearance_straight_m'] < plan['margin_m']:
        problems.append('straight-segment clearance is below the margin')
    return wp, problems


def print_table(args, wp, plan):
    print(f'plan {args.plan} (created {plan.get("created")}), leg {args.leg}, speed {args.vel_deg_s} deg/s, '
          f'accel {args.acc_deg_s2} deg/s^2')
    print(f'cube in base [m]: {np.round(plan["cube_centre_m"], 4).tolist()}   margin {plan["margin_m"] * 1000:.0f} mm, '
          f'cell {plan.get("cell")}')
    for k, w in enumerate(wp):
        turn = '' if k == 0 else f'   move {k}: largest joint turn {np.abs(w - wp[k - 1]).max():5.1f} deg, ' \
                                 f'~{np.abs(w - wp[k - 1]).max() / args.vel_deg_s:5.0f} s'
        print(f'  {k:2d} ' + ' '.join(f'{v:8.2f}' for v in w) + turn)


def main():
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    wp, problems = validate(args, plan)
    print_table(args, wp, plan)
    if problems:
        print('\nREFUSED:\n  - ' + '\n  - '.join(problems))
        sys.exit(2)
    print('\nCHECK OK.' + ('' if args.execute else '  (check only: nothing is sent; add --execute to move)'))
    if not args.execute:
        return

    import rclpy
    from dsr_msgs2.srv import GetCurrentPosj, MoveJoint, MoveStop
    rclpy.init()
    node = rclpy.create_node('supervised_pendant_mover')
    prefix = '/' + '/'.join(x for x in (args.robot_id, args.controller_ns) if x)
    get_posj = node.create_client(GetCurrentPosj, f'{prefix}/aux_control/get_current_posj')
    move_joint = node.create_client(MoveJoint, f'{prefix}/motion/move_joint')
    stop = node.create_client(MoveStop, f'{prefix}/motion/move_stop')
    for client in (get_posj, move_joint):
        if not client.wait_for_service(timeout_sec=10.0):
            sys.exit(f'service {client.srv_name} not available (bringup running? control granted to this PC?)')

    def call(client, request, timeout):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def current():
        res = call(get_posj, GetCurrentPosj.Request(), 3.0)
        if res is None or not res.success or len(res.pos) != 6:
            sys.exit('cannot read the current joints - stopping')
        return np.array(res.pos, dtype=float)

    def request_stop():
        try:
            req = MoveStop.Request()
            req.stop_mode = 1                       # DR_QSTOP: quick stop (never starts a motion)
            call(stop, req, 2.0)
        except Exception:
            pass

    print(f'\nThis will move the REAL robot, one confirmed move at a time, at {args.vel_deg_s} deg/s.')
    if input(f'Type exactly "{OPENING}" to continue: ').strip() != OPENING:
        sys.exit('aborted')
    try:
        for k in range(max(1, args.from_step), len(wp)):
            now = current()
            err = np.abs(now - wp[k - 1]).max()
            if err > args.max_start_error_deg:
                sys.exit(f'REFUSED before move {k}: robot is {err:.1f} deg from waypoint {k - 1} '
                         f'(> {args.max_start_error_deg}). Current {np.round(now, 1).tolist()}')
            turn = np.abs(wp[k] - now).max()
            print(f'\nmove {k}/{len(wp) - 1}: {np.round(now, 1).tolist()} -> {np.round(wp[k], 1).tolist()} '
                  f'(largest turn {turn:.1f} deg, ~{turn / args.vel_deg_s:.0f} s at {args.vel_deg_s} deg/s)')
            if input('type "go" to move, anything else aborts: ').strip() != 'go':
                sys.exit('aborted by user - robot left where it is')
            # A human may wait at the prompt or jog the robot on the pendant.
            now = current()
            if np.abs(now - wp[k - 1]).max() > args.max_start_error_deg:
                sys.exit('REFUSED: robot moved while waiting for go; make a new plan')
            _, problems = validate(args, plan)
            if problems:
                sys.exit('REFUSED: ' + '; '.join(problems))
            turn = np.abs(wp[k] - now).max()
            if turn > args.max_step_deg:
                sys.exit('REFUSED: actual joint turn exceeds the segment limit')
            req = MoveJoint.Request()
            req.pos = [float(v) for v in wp[k]]
            req.vel, req.acc, req.time, req.radius = float(args.vel_deg_s), float(args.acc_deg_s2), 0.0, 0.0
            req.mode, req.blend_type, req.sync_type = 0, 0, 0     # absolute, no blending, synchronous
            res = call(move_joint, req, turn / args.vel_deg_s * 2 + 20)
            if res is None or not res.success:
                request_stop()
                sys.exit('controller did not accept / complete the move (control authority on the pendant?) - '
                         'stopped, nothing retried')
            reached = np.abs(current() - wp[k]).max()
            print(f'  reached, max joint error {reached:.2f} deg')
            if reached > args.max_reach_error_deg:
                sys.exit(f'STOP: reached pose is {reached:.2f} deg off the target (> {args.max_reach_error_deg})')
        print('\nleg finished.')
    except KeyboardInterrupt:
        request_stop()
        print('\ninterrupted - stop requested')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
