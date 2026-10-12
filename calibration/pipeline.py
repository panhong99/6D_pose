"""One-terminal cube pipeline (conda env `foundationpose`).  Default: perception -> plan -> ROS publication, NO motion.

  bash calibration/run.sh connect                     terminal 1: Doosan bringup (Human)
  bash calibration/run.sh pipeline [--serial S]       terminal 2: this file (plan only)
  bash calibration/run.sh pipeline --execute          human-supervised real motion, one typed `go` per waypoint
  bash calibration/run.sh pipeline --offline-test     synthetic pose -> plan -> mover validation; no ROS / camera
  bash calibration/run.sh pipeline --task plate [--execute]   029 plate: FoundationPose -> BC+RL policy rollout (Isaac)
                                                      -> legs to_ready/grasp/lift/lower/retreat/to_home + gripper.py

D455 -> real_time_project/main.py -> UDP 5005 (FP pose) -> StablePose (n agreeing poses in the base frame)
ROS read-only joints (`ros2_nodes.py flange`) -> UDP 5006 -> start pose (must be < 1 s old; never zero-filled)
T_base_cam @ T_cam_object -> top-down pregrasp (cube yaw mod 90) -> IK -> clearance to the measured cell + D455 body
-> straight / RRT joint path, simplified to <= max_segment_deg point-to-point moves (approach and return legs)
-> plan_001.json/csv -> UDP 5010 -> `ros2_nodes.py publisher` (/sixd/dryrun/*, observation only)
-> (--execute) `ros2_nodes.py mover` per leg: slowest speed, re-checks the robot before every move, `go` per waypoint.
Outputs: data/real_pipeline/<time>/ (plan + one log per child process); offline runs in data/pipeline_offline/.
Numbers: config.yaml (planner, stability, udp, robot).  The mover's speed limits are constants in ros2_nodes.py.
"""
import argparse
import collections
import csv
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ros2_nodes as nodes
from calib_core import DATA, check_T, cpath, inv_T, latest_t_base_cam, load_config, load_json, rot_angle_deg, save_json
from robot_model import HOME_JOINTS_DEG, Kinematics, build_checker, cube_yaw_deg, grasp_frames, polyline_path
from plate_plans import (export_plan, isaac_cmd, isaac_env, full_plan, golden_slot_plan, library_ok_radii, library_target, next_full, next_slot,
                         plan_from_library, set_next_full, set_next_slot, slot_plan, teach_full, teach_slot)
from sensors import Receiver, RobotRecorder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C = load_config()


# ================================================================ planning (pure: no I/O)

def planner_params(**overrides):
    """config planner + stability as attributes (stable_n / stable_mm / stable_deg), with overrides."""
    p = dict(C['planner'])
    p.update(stable_n=C['stability']['n'], stable_mm=C['stability']['mm'], stable_deg=C['stability']['deg'])
    p.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**p)


def split_long_moves(waypoints, max_deg):
    """Insert evenly spaced points on any straight joint move turning a joint more than max_deg (same path, smaller
    steps).  simplify() keeps a direct start->goal pair even when it is long: home -> pregrasp turns j6 ~86 deg, which
    the mover refuses (> 45 deg per move) - found 2026-10-06 when the offline test started from the real home pose."""
    out = [np.asarray(waypoints[0], dtype=float)]
    for b in waypoints[1:]:
        a, b = out[-1], np.asarray(b, dtype=float)
        n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / max_deg - 1e-9)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


def plan_dryrun(kin, checker, T_base_obj, start_q, p):
    """Everything for one stable object pose; returns a result dict (never raises for 'bad' poses)."""
    centre = T_base_obj[:3, 3]
    yaw, tilt = cube_yaw_deg(T_base_obj)
    res = dict(dry_run=True, T_base_object=T_base_obj.tolist(), cube_centre_m=centre.tolist(), cube_yaw_deg=yaw,
               cube_tilt_deg=tilt, start_joints_deg=np.degrees(start_q).tolist(), margin_m=checker.margin,
               pregrasp_height_m=p.pregrasp_height_m, accepted=False, reasons=[])
    dist = float(np.linalg.norm(centre[:2]))
    if not p.reach_m[0] <= dist <= p.reach_m[1]:
        res['reasons'].append(f'object {dist:.2f} m from the base is outside {p.reach_m}')
    if not p.z_m[0] <= centre[2] <= p.z_m[1]:
        res['reasons'].append(f'object height z {centre[2]:.3f} m is outside {p.z_m} (wrong hand-eye / pose?)')
    if tilt > 10:
        res['reasons'].append(f'cube tilt {tilt:.1f} deg: no flat face is up')
    if res['reasons']:
        return res
    try:
        _, target, _, goal, q_grasp = grasp_frames(kin, centre, yaw, p.tool_length_m, p.pregrasp_height_m)
    except RuntimeError as exc:
        res['reasons'].append(str(exc))
        return res
    res.update(T_base_flange_pregrasp=target.tolist(), q_pregrasp_deg=np.degrees(goal).tolist(),
               q_grasp_deg_ik_only=np.degrees(q_grasp).tolist(),
               pregrasp_clearance_m=checker.clearance(goal), start_clearance_m=checker.clearance(start_q))
    if res['pregrasp_clearance_m'] < checker.margin:
        res['reasons'].append(f'pregrasp is only {res["pregrasp_clearance_m"] * 1000:.0f} mm from the cell '
                              f'(< {checker.margin * 1000:.0f} mm): move the object or raise the pregrasp height')
        return res
    if res['start_clearance_m'] < checker.margin:
        res['reasons'].append(f'start pose is only {res["start_clearance_m"] * 1000:.0f} mm from the cell')
        return res
    legs = {}
    for name, a, b in (('approach', start_q, goal), ('return', goal, start_q)):
        wp, how = checker.plan(a, b)
        wp = split_long_moves(checker.simplify(wp, p.max_segment_deg, checker.margin * 1.15), p.max_segment_deg)
        legs[name] = dict(planner=how, waypoints_deg=[np.degrees(w).round(3).tolist() for w in wp],
                          min_clearance_straight_m=float(checker.path_clearance(polyline_path(wp, 0.25))))
    res['legs'] = legs
    res['accepted'] = all(leg['min_clearance_straight_m'] >= checker.margin for leg in legs.values())
    if not res['accepted']:
        res['reasons'].append('straight-segment path comes closer than the margin to the cell')
    return res


def report(res):
    print('\n' + '=' * 78)
    print(f'cube in base [m]: {np.round(res["cube_centre_m"], 4).tolist()}  yaw {res["cube_yaw_deg"]:+.1f} deg  '
          f'tilt {res["cube_tilt_deg"]:.1f} deg')
    if 'q_pregrasp_deg' in res:
        print(f'pregrasp flange z {res["T_base_flange_pregrasp"][2][3]:.3f} m, clearance {res["pregrasp_clearance_m"] * 1000:.0f} mm '
              f'(margin {res["margin_m"] * 1000:.0f} mm)\nq pregrasp [deg]: {np.round(res["q_pregrasp_deg"], 1).tolist()}')
    for name, leg in res.get('legs', {}).items():
        print(f'\n{name}: {leg["planner"]}, {len(leg["waypoints_deg"])} waypoints, straight-segment min clearance '
              f'{leg["min_clearance_straight_m"] * 1000:.0f} mm')
        for k, w in enumerate(leg['waypoints_deg']):
            step = '' if k == 0 else '   turn max %5.1f deg' % np.abs(np.array(w) - np.array(leg['waypoints_deg'][k - 1])).max()
            print(f'  {k:2d}  ' + '  '.join(f'{v:8.2f}' for v in w) + step)
    print('\nRESULT:', 'ACCEPTED (dry run, nothing sent)' if res['accepted'] else 'REFUSED - ' + '; '.join(res['reasons']))
    print('=' * 78, flush=True)


def fresh_joints(msg, now=None):
    now = time.time() if now is None else now
    try:
        q = np.asarray(msg['posj'], dtype=float)
        age = now - float(msg['t'])
        return q.shape == (6,) and np.isfinite(q).all() and 0 <= age < 1.0
    except (KeyError, TypeError, ValueError):
        return False


class StablePose:
    """FP poses -> base frame; returns the median-translation pose once the last stable_n agree (else None)."""

    def __init__(self, T_base_cam, p):
        self.T_base_cam, self.p = T_base_cam, p
        self.history = collections.deque(maxlen=p.stable_n)
        self.last_stamp = 0

    def add(self, msg, now=None):
        now = time.time() if now is None else now
        try:
            stamp = float(msg['timestamp']['sec']) + float(msg['timestamp']['nanosec']) / 1e9
            T = check_T(msg['pose'], 'pose')
            if not np.isfinite(stamp) or not 0 <= now - stamp < 2 or stamp <= self.last_stamp:
                raise ValueError('stale or repeated pose')
        except (KeyError, TypeError, ValueError, RuntimeError):
            self.history.clear()
            return None
        if self.last_stamp and stamp - self.last_stamp > 1.0:
            self.history.clear()
        self.last_stamp = stamp
        self.history.append(self.T_base_cam @ T)
        if len(self.history) < self.p.stable_n:
            return None
        ref = self.history[-1]
        if (max(np.linalg.norm(h[:3, 3] - ref[:3, 3]) * 1000 for h in self.history) > self.p.stable_mm
                or max(rot_angle_deg(h[:3, :3].T @ ref[:3, :3]) for h in self.history) > self.p.stable_deg):
            return None
        result = ref.copy()
        result[:3, 3] = np.median([h[:3, 3] for h in self.history], axis=0)
        return result


def make_plan(kin, checker, T, q_deg, p, t_path, cell_json):
    try:
        res = plan_dryrun(kin, checker, T, np.radians(q_deg), p)
    except RuntimeError as exc:
        raise RuntimeError(f'path planning refused: {exc}') from exc
    res.update(t_base_cam_file=str(t_path), cell=str(cell_json), created=time.strftime('%Y-%m-%d %H:%M:%S'),
               joint_source='live_ros2', pipeline='pipeline.py')
    if res['accepted']:                              # the mover's own checks must pass too
        problems = [msg for leg in ('approach', 'return')
                    for msg in nodes.validate(nodes.mover_args(['--plan', 'unused', '--leg', leg]), res)[1]]
        if problems:
            res['accepted'] = False
            res['reasons'].extend(problems)
    return res


def save_plan(out, res):
    path = out / 'plan_001.json'
    save_json(path, res)
    with (out / 'plan_001.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['leg', 'step', *[f'j{i}_deg' for i in range(1, 7)]])
        for name, leg in res.get('legs', {}).items():
            for i, w in enumerate(leg['waypoints_deg']):
                writer.writerow([name, i, *w])
    report(res)
    print(f'Plan: {path}', flush=True)
    return path


# ================================================================ processes

def ros_environment(args):
    """Source ROS for the system-Python children without leaking conda's Python ABI."""
    for path in (args.ros_setup, args.doosan_setup):
        if not path.is_file():
            raise RuntimeError(f'ROS setup file missing: {path}')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH'):
        env.pop(key, None)
    env['PATH'] = '/usr/bin:/bin'
    res = subprocess.run(['/bin/bash', '-c', 'source "$1" >&2 && source "$2" >&2 && exec /usr/bin/env -0',
                          'ros-env', str(args.ros_setup), str(args.doosan_setup)],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
    return dict(item.decode().split('=', 1) for item in res.stdout.split(b'\0') if item)


def ros_node_command(mode, *extra):
    return ['/usr/bin/python3', '-u', str(HERE / 'ros2_nodes.py'), mode, *[str(e) for e in extra]]


def mover_command(args, path, leg, go_per='waypoint', opening_done=False, smooth=False, vel=None):
    cmd = ros_node_command('mover', '--plan', path, '--leg', leg, '--robot_id', args.robot_id, '--controller_ns', args.controller_ns,
                           '--go_per', go_per, *(['--opening_done'] if opening_done else []),
                           *(['--smooth', '--auto'] if smooth else []),
                           *(['--vel_deg_s', vel, '--acc_deg_s2', vel] if vel else []))
    if args.execute:
        cmd.append('--execute')
    return cmd


def _die_with_parent():
    """(10/12: the pipeline segfaulted twice and left perception + streams holding the cameras) background children
    get SIGTERM when this process dies.  Not used for the mover, which must finish / stop its own leg."""
    import ctypes
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)     # PR_SET_PDEATHSIG


class Children:
    """Only terminate subprocesses owned by this invocation, including on failures."""

    def __init__(self, out):
        self.out, self.items, self.active_mover = out, [], None

    def start(self, name, command, env=None):
        log = self.out / f'{name}.log'
        with log.open('wb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent)
        self.items.append((name, child, log))
        return child

    def check(self):
        for name, child, log in self.items:
            if child.poll() is not None:
                tail = '\n'.join(log.read_text(errors='replace').splitlines()[-12:])
                raise RuntimeError(f'{name} exited ({child.returncode}). Log: {log}\n{tail}')

    def wait_file(self, path, timeout=15):
        deadline = time.monotonic() + timeout
        while True:
            self.check()
            if path.exists():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f'timed out waiting for {path.name}; check {self.out}')
            time.sleep(0.1)

    def run_mover(self, command, env):
        self.check()
        self.active_mover = subprocess.Popen(command, cwd=ROOT, env=env)       # same process group: Ctrl+C reaches it
        while self.active_mover.poll() is None:
            self.check()
            time.sleep(0.1)
        code = self.active_mover.returncode
        self.active_mover = None
        if code:
            raise RuntimeError(f'mover stopped (exit {code}); no automatic retry or return')

    def close(self):
        if self.active_mover is not None and self.active_mover.poll() is None:
            self.active_mover.send_signal(signal.SIGINT)
            try:
                self.active_mover.wait(timeout=8)
            except subprocess.TimeoutExpired:
                # Never force-kill a process that may be responsible for stopping motion.
                print('Mover stop not confirmed. Check pendant / physical emergency stop.', flush=True)
        for _, child, _ in reversed(self.items):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for _, child, _ in reversed(self.items):
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def wait_choice(children):
    print('\n도착 완료. 복귀하려면 return, 현재 위치에서 종료하려면 quit 입력: ', end='', flush=True)
    while True:
        children.check()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            return sys.stdin.readline().strip() == 'return'


# ================================================================ run

def run(args):
    t_path = (args.t_base_cam or latest_t_base_cam()).resolve()
    Tbc = check_T(load_json(t_path)['T_base_cam'], f'T_base_cam {t_path}')
    p = planner_params(tool_length_m=args.tool_length_m, tool_radius_m=args.tool_radius_m,
                       pregrasp_height_m=args.pregrasp_height_m)
    kin = Kinematics()
    checker, _ = build_checker(kin, Tbc, p.margin_m, p.tool_length_m, p.tool_radius_m, args.cell_json)
    parent = DATA / ('pipeline_offline' if args.offline_test else 'real_pipeline')
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_'), dir=parent))
    print(f'Mode: {"OFFLINE" if args.offline_test else "SUPERVISED" if args.execute else "PLAN ONLY"}\n'
          f'T_base_cam: {t_path}\nCell: {args.cell_json}\nLogs: {out}', flush=True)
    stable = StablePose(Tbc, p)
    if args.offline_test:
        Tcam = inv_T(Tbc) @ np.array([[1, 0, 0, 1.04], [0, 1, 0, 0.05], [0, 0, 1, 0.0285], [0, 0, 0, 1.0]])
        now, T = time.time(), None
        for i in range(p.stable_n):
            stamp = now - 0.2 + i * 0.01
            T = stable.add(dict(pose=Tcam.tolist(), timestamp=dict(sec=int(stamp), nanosec=int((stamp % 1) * 1e9))), now=now)
        if T is None:
            raise RuntimeError('offline pose stability check failed')
        res = make_plan(kin, checker, T, HOME_JOINTS_DEG, p, t_path, args.cell_json)
        res.update(joint_source='synthetic_offline', offline_test=True, execution_allowed=False)   # never a real-motion plan
        save_plan(out, res)
        if not res['accepted']:
            raise RuntimeError('offline plan refused: ' + '; '.join(res['reasons']))
        print('OFFLINE OK: transform -> stability -> IK/collision -> approach/return validation. No ROS/camera.', flush=True)
        return

    udp = C['udp']
    children, receivers = Children(out), []
    try:
        env = ros_environment(args)
        for port in (udp['fp_pose'], udp['flange']):
            receivers.append(Receiver(port))
        poses, joints = receivers
        children.start('robot_state', ros_node_command('flange', '--robot_id', args.robot_id, '--controller_ns',
                                                       args.controller_ns, '--port', udp['flange']), env)
        ready, ack = out / 'publisher.ready', out / 'publisher.ack'
        children.start('ros_publisher', ros_node_command('publisher', '--port', udp['publish'], '--ready-file', ready,
                                                         '--ack-file', ack), env)
        children.wait_file(ready)
        print('[1/4] ROS 연결 확인: 현재 로봇 관절값을 기다립니다.', flush=True)
        deadline, cur = time.monotonic() + 20, None
        while not fresh_joints(cur):
            children.check()
            for cur in joints.packets():
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('no fresh robot joints; start terminal 1 and check robot_state.log')
            time.sleep(0.05)
        vision_env = dict(os.environ)
        vision_env.pop('PYTHONPATH', None)                    # conda inference does not need ROS's packages
        vision_env.pop('PYTHONHOME', None)
        children.start('perception', [sys.executable, '-u', str(ROOT / 'real_time_project/main.py'), '--detector', 'yoloe',
                                      '--width', '640', '--height', '480', '--serial', args.serial, '--prompt', args.prompt,
                                      '--conf', str(args.conf)], vision_env)
        print('[2/4] 인식 창에서 큐브를 확인하세요. 안정된 pose를 얻으면 자동으로 경로를 계산합니다.', flush=True)
        deadline = time.monotonic() + args.timeout
        while True:
            children.check()
            for msg in joints.packets():
                cur = msg
            for msg in poses.packets():
                T = stable.add(msg)
                if T is not None and fresh_joints(cur):
                    res = make_plan(kin, checker, T, cur['posj'], p, t_path, args.cell_json)
                    path = save_plan(out, res)
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as pub:
                        pub.sendto(json.dumps(res, allow_nan=False).encode(), ('127.0.0.1', udp['publish']))
                    children.wait_file(ack)
                    if not res['accepted']:
                        raise RuntimeError('경로 거부: ' + '; '.join(res['reasons']))
                    break
            else:
                if time.monotonic() > deadline:
                    raise RuntimeError('stable pose/live joints timeout; check recognition window and perception.log')
                time.sleep(0.05)
                continue
            break
        print('[3/4] 경로 저장 및 ROS /sixd/dryrun/* 발행 완료.', flush=True)
        if not args.execute:
            print('[4/4] 계획 검사 완료. 로봇 이동 없이 종료합니다.', flush=True)
            return
        print('[4/4] 같은 터미널에서 접근 승인을 진행합니다. 목표는 방금 저장한 큐브 위치입니다.', flush=True)
        children.run_mover(mover_command(args, path, 'approach'), env)
        if wait_choice(children):
            children.run_mover(mover_command(args, path, 'return'), env)
    finally:
        children.close()
        for receiver in receivers:
            receiver.sock.close()


# ================================================================ plate task (RL policy rollout, 2026-10-07)

def plate_root_in_base(T_base_obj):
    """Sim plate root (mesh bounding-box bottom centre) from a FoundationPose mesh pose in base."""
    import trimesh
    lo, hi = trimesh.load(cpath(C['paths']['plate_obj']), force='mesh').bounds
    p = (T_base_obj @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2], 1.0]))[:3]
    yaw = float(np.degrees(np.arctan2(T_base_obj[1, 0], T_base_obj[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(T_base_obj[2, 2], -1, 1))))
    return p, yaw, tilt


def rl_target():
    import yaml
    g = yaml.safe_load((ROOT / 'rl' / 'config.yaml').read_text())['geometry']
    return np.array([*g['plate_xy'], g['table_top_z'] + g['box_h']])


def plate_cfg(policy):
    """plate_rl, or plate_rl overridden by plate_ik_bc (--plate_policy ik_bc: IK approach + BC grasp, wide area)."""
    return {**C['plate_rl'], **C['plate_ik_bc']} if policy == 'ik_bc' else dict(C['plate_rl'])


def run_export_server(args):
    """Foreground warm Isaac export server for the selected plate policy (Ctrl+C stops it).  Start it once, in its
    own terminal, before the pipeline runs; restart it after changing the plate_rl / plate_ik_bc checkpoint."""
    q = plate_cfg(args.plate_policy)
    cmd = isaac_cmd(q, 'serve') + ['--port', str(C['udp'].get('export_server', 5012))]
    print('export server:', ' '.join(cmd), flush=True)
    while True:                                             # Isaac start-up segfaults: start again
        code = subprocess.run(cmd, cwd=ROOT, env=isaac_env()).returncode
        if code in (0, -signal.SIGINT, 130):
            return
        print(f'export server exited ({code}); restarting in 3 s', flush=True)
        time.sleep(3)


_VIZ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def viz(**msg):
    """Fire-and-forget JSON to `run.sh viz` (ros2_nodes.py viz, UDP udp.viz); nothing happens when it is not running."""
    try:
        _VIZ.sendto(json.dumps(msg).encode(), ('127.0.0.1', C['udp'].get('viz', 5011)))
    except OSError:
        pass


def gripper(cmd, env, check=False):
    """gripper.py runs in system Python (pyserial); motion commands ask the Human for `yes` themselves."""
    full = ['/usr/bin/python3', str(HERE / 'gripper.py'), *cmd]
    if check:
        return subprocess.run(full, cwd=ROOT, env=env, capture_output=True, text=True, timeout=10).stdout
    if subprocess.run(full, cwd=ROOT, env=env).returncode:
        raise RuntimeError(f'gripper {" ".join(cmd)} failed or was cancelled')


def run_plate(args):
    """FoundationPose plate -> BC+RL rollout plan -> supervised legs with the gripper in between.
    Legs: to_ready, grasp, [close + object check], lift, [Human: lower?], lower, [open], retreat, to_home."""
    q, udp = plate_cfg(args.plate_policy), C['udp']
    if args.no_library:                                     # the 10/8 validated mode: sim rollout per plate (export)
        q.pop('library', None)
    t_path = (args.t_base_cam or latest_t_base_cam()).resolve()
    Tbc = check_T(load_json(t_path)['T_base_cam'], f'T_base_cam {t_path}')
    out = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_plate_'), dir=DATA / 'real_pipeline'))
    import faulthandler                                     # (10/12) two segfaults: keep every thread's stack
    faulthandler.enable(open(out / 'faulthandler.log', 'w'), all_threads=True)
    target, tol = rl_target(), np.array(q['tolerance_mm'])
    lo, hi = -tol, tol                                      # allowed (FP + offset - trained spot) box, mm
    if q.get('workspace_mm'):
        w = q['workspace_mm']
        lo, hi = np.array([w['x'][0], w['y'][0], w['z'][0]]), np.array([w['x'][1], w['y'][1], w['z'][1]])
    print(f'Mode: PLATE ({args.plate_policy}) {"SUPERVISED" if args.execute else "PLAN ONLY"}\nT_base_cam: {t_path}\n'
          f'RL target root: {np.round(target, 3).tolist()} (allowed mm x {lo[0]:.0f}..{hi[0]:.0f}, '
          f'y {lo[1]:.0f}..{hi[1]:.0f})\nLogs: {out}', flush=True)
    off3 = np.array([*q['offset_m'], 0.0])
    area = [(target - off3 + lo / 1000).tolist(), (target - off3 + hi / 1000).tolist()]   # plate-root box, base frame
    area[0][2], area[1][2] = target[2] - 0.003, target[2] + 0.003                        # thin floor patch
    viz(area=area)
    st = C['stability']
    children, receivers = Children(out), []
    try:
        env = ros_environment(args)
        if args.execute:
            receivers.append(Receiver(udp['flange']))
            children.start('robot_state', ros_node_command('flange', '--robot_id', args.robot_id, '--controller_ns',
                                                           args.controller_ns, '--port', udp['flange']), env)
            print('[1/5] 로봇 관절값 확인 (홈 자세여야 함)', flush=True)
            deadline, cur = time.monotonic() + 20, None
            while not fresh_joints(cur):
                children.check()
                for cur in receivers[0].packets():
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('no fresh robot joints; start the bringup (run.sh connect)')
                time.sleep(0.05)
            err = np.abs(np.array(cur['posj']) - np.array(HOME_JOINTS_DEG)).max()
            if err > q['start_tolerance_deg']:
                raise RuntimeError(f'robot is {err:.1f} deg from home; move it to home with the pendant first')
        basket_delta, basket_T = None, None
        if args.place == 'basket':                           # markers first: the fixed D455 is FoundationPose's afterwards
            from camera_tools import basket_fit_robust, basket_measure
            slot1 = json.loads((DATA / 'basket' / 'slot_01_in_basket.json').read_text())
            layout = json.loads(cpath(C['basket']['layout']).read_text())
            Tb, res, seen = basket_fit_robust(basket_measure(C), layout, C['basket'].get('max_fit_residual_m', 0.008))
            T0 = np.array(slot1['T_base_basket_at_teach'])
            basket_delta = Tb @ np.linalg.inv(T0)
            basket_T = Tb
            mv = (Tb[:3, 3] - T0[:3, 3]) * 1000
            yaw = np.degrees(np.arctan2(basket_delta[1, 0], basket_delta[0, 0]))
            print(f'[basket] markers {seen}, fit {res * 1000:.1f} mm: moved x {mv[0]:+.0f} y {mv[1]:+.0f} mm, '
                  f'yaw {yaw:+.1f} deg since the teaching', flush=True)
            if res > C['basket'].get('max_fit_residual_m', 0.008) or len(seen) < 3:
                raise RuntimeError(f'basket pose unreliable: {len(seen)} markers, residual {res * 1000:.1f} mm')
        from sensors import FpStream
        fp = FpStream(udp['fp_pose'], st['n'], st['mm'], st['deg'], symmetric_z=True)   # round plate: yaw is noise
        vision_env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        perception = children.start('perception', [sys.executable, '-u', str(ROOT / 'real_time_project/main.py'),
                                                    '--detector', 'yoloe', '--width', '640', '--height', '480',
                                                    '--serial', args.serial, '--prompt', q['prompt'], '--conf', str(args.conf),
                                                    '--mesh_file', str(cpath(C['paths']['plate_obj'])),
                                                    '--wrist_serial', C['cameras']['wrist_serial']], vision_env)
        if q.get('library'):
            rlo, rhi = library_ok_radii(q)
            rlo, rhi = rlo + 0.005, rhi - 0.005
            print(f'[2/5] 접시 인식 중. 로봇에서 {rlo:.2f}~{rhi:.2f} m, 방향 +-{q["max_azimuth_deg"]:.0f} deg 안이면 '
                  f'자동으로 다음 단계로 갑니다.', flush=True)
        else:
            print('[2/5] 접시 인식 중. 학습 위치 범위 안에 놓이면 자동으로 다음 단계로 갑니다.', flush=True)
        deadline, last, good = time.monotonic() + max(args.timeout, q['placement_timeout_s']), 0.0, 0
        while True:
            children.check()
            fp.poll()
            T = fp.stable()
            if T is not None and time.time() - last > 0.5:
                last = time.time()
                p, yaw, tilt = plate_root_in_base(Tbc @ T)
                d = (p + np.array([*q['offset_m'], 0.0]) - target) * 1000     # the rollout uses FP + offset
                hint = []
                if q.get('library'):                                           # any direction, radius band
                    r, az = library_target(p, q)
                    ok = rlo <= r <= rhi and abs(np.degrees(az)) <= q['max_azimuth_deg'] and tilt <= q['max_tilt_deg']
                    if r < rlo or r > rhi:
                        hint.append(f'{"로봇에서 멀리" if r < rlo else "로봇 쪽으로"} {abs(r - np.clip(r, rlo, rhi)) * 100:.1f} cm')
                    if abs(np.degrees(az)) > q['max_azimuth_deg']:
                        hint.append(f'정면 쪽으로 (방향 {np.degrees(az):+.0f} deg)')
                    d[:2] = [(r - (rlo + rhi) / 2) * 1000, np.degrees(az)]
                else:
                    ok = ((d >= lo) & (d <= hi)).all() and tilt <= q['max_tilt_deg']
                viz(plate=(Tbc @ T).tolist(), ok=bool(ok), area=None if q.get('library') else area)
                good = good + 1 if ok else 0
                if not q.get('library') and not lo[0] <= d[0] <= hi[0]:
                    over = d[0] - hi[0] if d[0] > hi[0] else d[0] - lo[0]
                    hint.append(f'{"로봇 쪽으로" if over > 0 else "로봇에서 멀리"} {abs(over) / 10:.1f} cm')
                if not q.get('library') and not lo[1] <= d[1] <= hi[1]:
                    over = d[1] - hi[1] if d[1] > hi[1] else d[1] - lo[1]
                    hint.append(f'{"오른쪽으로" if over > 0 else "왼쪽으로"} {abs(over) / 10:.1f} cm')
                if tilt > q['max_tilt_deg']:
                    hint.append(f'기울기 {tilt:.1f} deg (평평하게)')
                where = (f'r {r:.3f} m, 방향 {np.degrees(az):+.0f} deg' if q.get('library')
                         else f'dx {d[0]:+.0f} dy {d[1]:+.0f} dz {d[2]:+.0f} mm')
                print(f'   plate root {np.round(p, 4).tolist()}  ' + ('OK' if ok else '옮기기: ' + ', '.join(hint)) +
                      f'   ({where}, tilt {tilt:.1f})', flush=True)
                if good >= 4:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError('no stable plate pose inside the trained area before the timeout')
            time.sleep(0.05)
        if not q.get('keep_perception', True):
            children.items = [it for it in children.items if it[1] is not perception]
            os.killpg(perception.pid, signal.SIGINT)            # free the GPU for the Isaac rollout
            perception.wait(timeout=20)
        if q.get('library'):
            r, az = library_target(p, q)
            x, y = r * np.cos(az), r * np.sin(az)
            print(f'[3/5] 경로 라이브러리: r {r:.3f} m, 방향 {np.degrees(az):+.1f} deg (joint 1 회전)', flush=True)
            plan = plan_from_library(r, az, q, Tbc, plate_z=float(p[2]))
        else:
            x, y = p[0] + q['offset_m'][0], p[1] + q['offset_m'][1]
            print(f'[3/5] BC+RL 정책 rollout (sim) at ({x:.4f}, {y:.4f}) yaw {yaw:.1f}: 1~2 분', flush=True)
            plan = export_plan(x, y, yaw, out / 'export.log', q)
        early, shown = None, plan
        if args.place in ('golden', 'basket'):          # (10/12) whole path before moving: reach check + RViz preview
            legs0 = json.loads(Path(plan).read_text())['legs']
            early = golden_slot_plan(q['golden_teach'], legs0['to_grasp']['waypoints_deg'][-1], legs0['lift']['waypoints_deg'][-1],
                                     Tbc, q, basket_delta=basket_delta, plate_xy=p[:2])
            both = json.loads(Path(plan).read_text())
            both['legs'].update(json.loads(early.read_text())['legs'])
            shown = out / 'preview_plan.json'
            shown.write_text(json.dumps(dict(both, preview_only=True)))
            print(f'   바구니 경로까지 계획 완료 (RViz 미리보기: 잡기 + 바구니 전체): {early}', flush=True)
        viz(plan=str(shown))
        print(f'   plan: {plan}  (30 분 유효)', flush=True)
        if not args.execute:
            print('[5/5] 계획만 만들었습니다. --execute 로 실제 이동.', flush=True)
            return
        vel = q.get('vel_deg_s', 2.0)

        def leg(name):
            print(f'\n=== {name}: one smooth spline at {vel} deg/s ===', flush=True)
            rec.event('leg_start', leg=name, vel_deg_s=vel, smooth=True)
            children.run_mover(mover_command(args, plan, name, 'leg', opening_done=True, smooth=True, vel=vel), env)
            rec.event('leg_end', leg=name)

        def choose_gripper(where):
            # the Human decides every gripper action (--auto: fixed points only, robot standing still, see below)
            while not args.auto:
                ans = input(f'\n[{where}] 그리퍼: c = 닫기, o = 열기, s = 그대로 두고 진행 > ').strip().lower()
                if ans in ('c', 'o', 's'):
                    break
            if args.auto:
                ans = 'c' if where == 'grasp' else 'o'
                if ans == 'c':                              # the plate must still be where the plan was made for
                    now = fp_sample()
                    moved = None if not now else 1000 * float(np.hypot(*np.subtract(now['plate_root'][:2], p[:2])))
                    print(f'[auto] 잡기 전 접시 위치 확인: FP 이동 {moved if moved is None else round(moved, 1)} mm', flush=True)
                    if moved is not None and moved > q.get('auto_max_plate_move_mm', 15.0):
                        ans = 's'
                print(f'[auto] 그리퍼 {where}: {dict(c="닫기", o="열기", s="닫지 않음 (접시가 움직임)")[ans]}', flush=True)
            if ans != 's':
                gripper(['close' if ans == 'c' else 'open', '--yes'], env)
            st = gripper(['status'], env, check=True)
            rec.event('gripper', where=where, choice=ans, status=st,
                      grasped='stopped on an object while closing' in st)
            print(st, flush=True)
            return ans, st
        print('[4/5] 실행: 시작 확인 문구 한 번, 이후 leg 마다 go 한 번 (leg 안의 waypoint 는 연속 진행)', flush=True)
        while input(f'\n실제 로봇이 움직입니다. "{nodes.OPENING}" 를 정확히 입력: ').strip() != nodes.OPENING:
            print('  문구가 다릅니다 (q 로 종료하려면 Ctrl+C)', flush=True)
        def fp_sample():                                    # live FoundationPose plate root in base (tracking view stays open)
            fp.poll()
            if not fp.hist:
                return None
            root, fyaw, ftilt = plate_root_in_base(Tbc @ fp.hist[-1])
            viz(plate=(Tbc @ fp.hist[-1]).tolist(), ok=True)
            return dict(plate_root=[round(float(v), 5) for v in root], yaw_deg=round(fyaw, 2), tilt_deg=round(ftilt, 2))
        rec = RobotRecorder(receivers[0], out / 'robot_state.jsonl', fp_sample)   # every ROS2 joint/flange sample + FP
        rec.event('plan', plan=str(plan), plate_root_fp=[float(v) for v in p], yaw_deg=yaw, tilt_deg=tilt,
                  offset_m=q['offset_m'], rollout_plate=[x, y])
        if 'status   : activated' not in gripper(['status'], env, check=True):
            print('그리퍼가 활성화되어 있지 않습니다 (로봇 재연결 시 전원이 다시 켜짐). 활성화합니다: '
                  '손가락이 한 번 끝까지 열고 닫힙니다.', flush=True)
            gripper(['activate'], env)
        choose_gripper('start')
        leg('to_grasp')                       # home -> ready -> rim, one smooth motion
        _, status = choose_gripper('grasp')
        if 'stopped on an object while closing' not in status:
            print('물체가 잡혀 있지 않습니다. 들지 않고 되돌아갑니다 (그리퍼를 열고 싶으면 o).', flush=True)
            choose_gripper('abort')
            leg('from_grasp')
            raise RuntimeError('no grasp: measure how far the plate had to move and set plate_rl.offset_m')
        leg('lift')
        rec.event('lifted')
        time.sleep(q.get('hold_s', 3.0))
        lift_end = json.loads(Path(plan).read_text())['legs']['lift']['waypoints_deg'][-1]
        if args.place == 'teach':
            k = args.slot or next_slot()
            teach_slot(k, rec, choose_gripper, lift_end)
            rec.event('done', place='teach', slot=k)
            print('[5/5] 티칭 완료.', flush=True)
            return
        if args.place in ('slot', 'golden', 'basket'):
            k = args.slot or (1 if args.place in ('golden', 'basket') else next_slot())
            if args.place in ('golden', 'basket'):            # FP grasp + the Human's best taught basket path
                grasp_deg = json.loads(Path(plan).read_text())['legs']['to_grasp']['waypoints_deg'][-1]
                splan = early or golden_slot_plan(q['golden_teach'], grasp_deg, lift_end, Tbc, q, basket_delta=basket_delta, plate_xy=p[:2])
            else:
                splan = slot_plan(k, lift_end, Tbc, q)
            viz(plan=str(splan))
            rec.event('slot_plan', slot=k, plan=str(splan))

            cur = dict(plan=splan)                                  # the pre-stop correction swaps in a shifted plan

            def sleg(name):
                v = q.get('basket_insert_vel_deg_s', vel) if name in ('descend', 'shift') else vel   # in the pins: slow
                if name == 'ascend':                            # (10/12) released: the open fingers only leave the rim
                    v = q.get('basket_ascend_vel_deg_s', vel)
                if name in ('descend_pre', 'pre_up'):             # (10/12 Human) the first descent over the basket too fast
                    v = q.get('basket_pre_vel_deg_s', vel)
                print(f'\n=== slot {k} {name}: one smooth spline at {v} deg/s ===', flush=True)
                rec.event('leg_start', leg=f'slot{k}_{name}', vel_deg_s=v, smooth=True)
                children.run_mover(mover_command(args, cur['plan'], name, 'leg', opening_done=True, smooth=True, vel=v), env)
                rec.event('leg_end', leg=f'slot{k}_{name}')
            if 'to_above' in json.loads(Path(splan).read_text())['legs']:     # golden / basket: four stages
                def look(tag):                                      # read-only wrist photo + model pins (10/12)
                    try:                                            # child process: OpenCV stays out of this one
                        t_ask = time.time()                         # joints are not streamed during a move: wait for a fresh one
                        while (rec.last or {}).get('t', 0) < t_ask and time.time() - t_ask < 5:
                            time.sleep(0.05)
                        Tbb = basket_T if basket_T is not None else np.array(json.loads(
                            (DATA / 'basket' / 'slot_01_in_basket.json').read_text())['T_base_basket_at_teach'])
                        Tbb = Tbb.copy()
                        if basket_T is None:                        # golden: the basket where it was found today
                            Tbb[:3, 3] += q.get('basket_shift_m', [0.0, 0.0, 0.0])
                        png = out / f'wrist_{tag}.png'
                        req = dict(T_base_flange=Kinematics().fk(np.radians(rec.last['posj'])).tolist(), T_base_basket=Tbb.tolist(),
                                   png=str(png), after=t_ask)
                        (out / f'wrist_{tag}.req.json').write_text(json.dumps(req))
                        subprocess.run([sys.executable, str(HERE / 'camera_tools.py'), 'pre_photo', '--req',
                                        str(out / f'wrist_{tag}.req.json')], check=True, timeout=60)
                        rec.event('wrist_snapshot', at=tag, png=str(png))
                        print(f'   손목 사진: {png} (초록 = 모델 핀 위치)', flush=True)
                    except Exception as exc:                        # a photo never blocks the run
                        print(f'   손목 사진 실패: {exc}', flush=True)

                def ask(text, keys):
                    if args.auto:                               # (10/12) no Human answers: go on ('d')
                        print(text.strip() + ' [auto] d', flush=True)
                        return 'd'
                    while (a_ := input(text).strip().lower()) not in keys:
                        pass
                    return a_
                sleg('to_above')
                look('above')
                if args.stop_at_above:
                    print('[1/4] 바구니 위에서 멈춤 (--stop_at_above). 접시는 잡은 채 그대로입니다.', flush=True)
                    rec.event('done', place=args.place, slot=k, stopped='above')
                    return
                if ask('\n[1/4] 바구니 위. d = 2단계 하강 시작, h = 하강 없이 홈 복귀 > ', ('d', 'h')) == 'h':
                    sleg('abort_home')
                    rec.event('done', place=args.place, slot=k, stopped='above_home')
                    return
                sleg('descend_pre')
                look('pre')
                a_ = ask('\n[2/4] 넣기 직전(접시 바닥이 빗살 레일 약 4 cm 위). d = 넣기, u = 다시 올려 홈 복귀 > ', ('d', 'u'))
                if a_ == 'u':
                    sleg('pre_up')
                    sleg('abort_home')
                    rec.event('done', place=args.place, slot=k, stopped='pre_up')
                    return
                sleg('descend')
                choose_gripper(f'slot{k}_release')
                sleg('ascend')
                sleg('to_home')
            else:
                sleg('to_slot')
                choose_gripper(f'slot{k}_release')
                sleg('slot_out')
            if not args.slot and args.place == 'slot':
                set_next_slot(k + 1)
            rec.event('done', place='slot', slot=k)
            print(f'[5/5] 슬롯 {k} 완료. 다음 접시는 슬롯 {next_slot()}.', flush=True)
            return
        leg('lower')
        choose_gripper('release')
        leg('from_grasp')
        rec.event('done')
        print(f'[5/5] 완료. 로봇 기록: {out / "robot_state.jsonl"}', flush=True)
    finally:
        if 'rec' in locals():
            rec.stop()
        children.close()
        for r in receivers:
            r.sock.close()


def run_taught(args):
    """--task teach: Human jogs pick + basket slot keyposes (plate at a fixed marked spot), Enter stores joints.
    --task replay: those keyposes as smooth 6 deg/s legs (to_grasp | close | to_slot | open | slot_out), no vision."""
    q, udp = plate_cfg(args.plate_policy), C['udp']
    Tbc = check_T(load_json((args.t_base_cam or latest_t_base_cam()).resolve())['T_base_cam'], 'T_base_cam')
    k = args.slot or next_full()
    out = Path(tempfile.mkdtemp(prefix=time.strftime(f'%Y%m%d_%H%M%S_{args.task}{k:02d}_'), dir=DATA / 'real_pipeline'))
    children, receiver = Children(out), Receiver(udp['flange'])
    try:
        env = ros_environment(args)
        children.start('robot_state', ros_node_command('flange', '--robot_id', args.robot_id, '--controller_ns',
                                                       args.controller_ns, '--port', udp['flange']), env)
        rec = RobotRecorder(receiver, out / 'robot_state.jsonl')
        deadline = time.monotonic() + 20
        while not fresh_joints(rec.last):
            children.check()
            if time.monotonic() > deadline:
                raise RuntimeError('no fresh robot joints; start the bringup (run.sh connect)')
            time.sleep(0.05)

        def choose_gripper(where):
            while True:
                ans = input(f'\n[{where}] 그리퍼: c = 닫기, o = 열기, s = 그대로 > ').strip().lower()
                if ans in ('c', 'o', 's'):
                    break
            if ans != 's':
                gripper(['close' if ans == 'c' else 'open', '--yes'], env)
            st = gripper(['status'], env, check=True)
            rec.event('gripper', where=where, choice=ans, status=st, grasped='stopped on an object while closing' in st)
            print(st, flush=True)
            return ans, st
        if 'status   : activated' not in gripper(['status'], env, check=True):
            print('그리퍼 활성화 (Human 확인)', flush=True)
            gripper(['activate'], env)
        print(f'Mode: {args.task.upper()} {k}  logs {out}', flush=True)
        if args.task == 'teach':
            choose_gripper(f'teach{k}_start (열기 o 권장)')
            teach_full(k, lambda: rec.last, choose_gripper, rec.event)
            return
        err = np.abs(np.array(rec.last['posj']) - np.array(HOME_JOINTS_DEG)).max()
        if err > q['start_tolerance_deg']:
            raise RuntimeError(f'robot is {err:.1f} deg from home; move it to home with the pendant first')
        plan = full_plan(k, Tbc, q)
        viz(plan=str(plan))
        rec.event('plan', plan=str(plan), slot=k)
        if not args.execute:
            print(f'계획만 만들었습니다: {plan} (--execute 로 실제 이동)', flush=True)
            return
        vel = q.get('vel_deg_s', 2.0)
        while input(f'\n실제 로봇이 움직입니다. "{nodes.OPENING}" 를 정확히 입력: ').strip() != nodes.OPENING:
            print('  문구가 다릅니다 (q 로 종료하려면 Ctrl+C)', flush=True)

        def leg(name):
            print(f'\n=== {name}: one smooth spline at {vel} deg/s ===', flush=True)
            rec.event('leg_start', leg=name, vel_deg_s=vel, smooth=True)
            children.run_mover(mover_command(args, plan, name, 'leg', opening_done=True, smooth=True, vel=vel), env)
            rec.event('leg_end', leg=name)
        choose_gripper('start (열기 o)')
        leg('to_grasp')
        _, st = choose_gripper('grasp (닫기 c)')
        if 'stopped on an object while closing' not in st:
            print('물체가 잡혀 있지 않습니다. 열고 되돌아갑니다.', flush=True)
            choose_gripper('abort')
            raise RuntimeError('no grasp: the plate is not at the taught spot')
        leg('to_slot')
        choose_gripper(f'slot{k} release (열기 o)')
        leg('slot_out')
        if not args.slot:
            set_next_full(k + 1)
        rec.event('done', slot=k)
        print(f'완료. 다음 실행은 {next_full()}번 티칭 경로. 기록 {out / "robot_state.jsonl"}', flush=True)
    finally:
        if 'rec' in locals():
            rec.stop()
        children.close()
        receiver.sock.close()


def parse_args(argv=None):
    pl, rb = C['planner'], C['robot']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=('cube', 'plate', 'teach', 'replay'), default='cube',
                   help='cube: top-down planner (default); plate: FoundationPose plate -> BC+RL rollout -> legs + gripper')
    p.add_argument('--plate_policy', choices=('rl', 'ik_bc'), default='rl',
                   help='plate: rl = BC+RL at the trained spot (config plate_rl); ik_bc = IK approach + BC grasp, '
                        'plate anywhere in plate_ik_bc.workspace_mm')
    p.add_argument('--export_server', action='store_true',
                   help='plate: run the warm Isaac export server for --plate_policy (own terminal; plans in seconds)')
    p.add_argument('--no_library', action='store_true',
                   help='plate ik_bc: no plan library (10/8 13:xx validated mode: ready -> line -> BC rolled out in sim '
                        'per plate, box workspace); the library mode is not yet run on the real robot')
    p.add_argument('--place', choices=('lower', 'teach', 'slot', 'golden', 'basket'), default='basket',
                   help='plate, after the lift (10/12: default basket): lower = put it back; teach = Human jogs to the basket slot, '
                        'Enter stores carry / above / insert / retreat; slot = replay the taught slot path; '
                        'golden = the best full teaching (config plate_ik_bc.golden_teach) corrected to this grasp; '
                        'basket = golden + the basket pose from its 4 markers (the slot path follows a moved basket)')
    p.add_argument('--auto', action='store_true', help='plate: no Human answers after the opening phrase: gripper '
                   'closes at the grasp pose (if FP says the plate did not move) and opens at the insert; stops go on')
    p.add_argument('--stop_at_above', action='store_true',
                   help='golden / basket: stop after stage 1 (plate held above the basket; Human takes over)')
    p.add_argument('--slot', type=int, default=None, help='slot number (default: data/teach/next_slot.json, then +1)')
    p.add_argument('--execute', action='store_true', help='human-supervised real motion, one go per waypoint')
    p.add_argument('--offline-test', action='store_true', help='synthetic pose -> plan -> mover validation; no hardware')
    p.add_argument('--serial', default=C['cameras']['fixed_serial'], help='FIXED D455 (an empty serial would pick either camera)')
    p.add_argument('--t_base_cam', type=Path, help='default: calib_core.latest_t_base_cam()')
    p.add_argument('--cell_json', type=Path, default=cpath(C['paths']['cell_json']))
    p.add_argument('--tool_length_m', type=float, default=pl['tool_length_m'])
    p.add_argument('--tool_radius_m', type=float, default=pl['tool_radius_m'])
    p.add_argument('--pregrasp_height_m', type=float, default=pl['pregrasp_height_m'])
    p.add_argument('--conf', type=float, default=pl['conf'])
    p.add_argument('--prompt', default=pl['prompt'])
    p.add_argument('--timeout', type=float, default=pl['timeout_s'], help='seconds allowed to obtain a stable pose/plan')
    p.add_argument('--robot_id', default=rb['robot_id'])
    p.add_argument('--controller_ns', default=rb['controller_ns'])
    p.add_argument('--ros_setup', type=Path, default=cpath(C['paths']['ros_setup']))
    p.add_argument('--doosan_setup', type=Path, default=cpath(C['paths']['doosan_setup']))
    a = p.parse_args(argv)
    if a.offline_test and a.execute:
        p.error('--offline-test cannot be combined with --execute')
    for name in ('timeout', 'pregrasp_height_m', 'tool_radius_m', 'conf'):
        if not np.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f'--{name} must be positive and finite')
    if not np.isfinite(a.tool_length_m) or a.tool_length_m < 0 or a.conf > 1:
        p.error('tool length must be nonnegative; conf must be <= 1')
    if a.execute and not sys.stdin.isatty():
        p.error('--execute requires an interactive terminal')
    return a


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.task == 'plate' and args.export_server:
            run_export_server(args)
        elif args.task in ('teach', 'replay'):
            run_taught(args)
        else:
            run_plate(args) if args.task == 'plate' else run(args)
    except KeyboardInterrupt:
        print('\n파이프라인 종료 요청.', flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'\nSTOP: {exc}', file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
