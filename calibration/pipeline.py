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
                         merged_plan, plan_from_library, policy_plan, set_next_full, set_next_slot, slot_plan, teach_full, teach_slot)
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


def mover_command(args, path, leg, go_per='waypoint', opening_done=False, smooth=False, vel=None, tool_force_limit_n=None):
    cmd = ros_node_command('mover', '--plan', path, '--leg', leg, '--robot_id', args.robot_id, '--controller_ns', args.controller_ns,
                           '--go_per', go_per, *(['--opening_done'] if opening_done else []),
                           *(['--smooth', '--auto'] if smooth else []),
                           *(['--tool-force-limit-n', tool_force_limit_n] if tool_force_limit_n is not None else []),
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

_PLATE_BOUNDS = []


def plate_bounds():
    """Plate mesh bounding box (lo, hi), loaded once."""
    if not _PLATE_BOUNDS:
        import trimesh
        _PLATE_BOUNDS.append(trimesh.load(cpath(C['paths']['plate_obj']), force='mesh').bounds)
    return _PLATE_BOUNDS[0]


def plate_upright(T_base_obj):
    """(10/14 실기) FoundationPose sometimes returns the round plate upside down (z down, 'tilt 179 deg') from the far
    fixed camera although it lies right side up: turn such a pose 180 deg about the mesh x axis through the bounding-box
    centre (same volume, z up).  A plate within 90 deg of upright is returned unchanged."""
    T_base_obj = np.asarray(T_base_obj, float)
    if T_base_obj[2, 2] >= 0:
        return T_base_obj
    lo, hi = plate_bounds()
    c, F = np.eye(4), np.diag([1.0, -1.0, -1.0, 1.0])
    c[:3, 3] = (lo + hi) / 2
    return T_base_obj @ c @ F @ np.linalg.inv(c)


def plate_root_in_base(T_base_obj):
    """Sim plate root (mesh bounding-box bottom centre) from a FoundationPose mesh pose in base (upside-down poses
    turned upright first, see plate_upright)."""
    T_base_obj = plate_upright(T_base_obj)
    lo, hi = plate_bounds()
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


def load_basket_insert_policy(path):
    """Load the deterministic, CPU-only basket insertion policy for the opt-in real pipeline."""
    expert_dir = str(ROOT / 'rl' / 'basket')
    if expert_dir not in sys.path:
        sys.path.insert(0, expert_dir)
    import expert
    policy = expert.load_insert_policy(path)
    expert._CL_POL.update(policy=policy, det=True, dr=False, dr_full=False, noseen=True,
                          canon=True, seed=0, cache={})
    return expert


def _last_good_path():
    return DATA / 'basket' / 'last_good_basket.json'


def save_last_good_basket(cb, Tb, res, seen, now=None):
    """(10/19) the basket pose of a run where all 4 markers fit (reused while plates stand in the basket)."""
    _last_good_path().write_text(json.dumps(dict(T_base_basket=np.asarray(Tb).tolist(), fit_m=float(res),
                                                 markers=[int(i) for i in seen], t=time.time() if now is None else now,
                                                 layout=cb.get('layout'))))


def basket_last_good(cb, now=None):
    """(T, age s) of the last 4-marker basket pose if younger than reuse_last_good_min and from the same layout, else None."""
    p = _last_good_path()
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    age = (time.time() if now is None else now) - float(d['t'])
    if d.get('layout') != cb.get('layout') or not 0 <= age <= 60 * cb.get('reuse_last_good_min', 30):
        return None
    return np.array(d['T_base_basket'], float), age


def fixed_basket_z(Tb, cb):
    """(10/19 보스) the basket stands on the table and its height never changes (17.5 cm overall, tines 6 cm), but the
    marker fit put the frame at z 0.125-0.143 m run to run, and the insert went that much deeper.  With `fixed_z_m` the
    frame height is that value; a fit farther off than `fixed_z_tol_m` is refused (wrong markers, not noise)."""
    fz = cb.get('fixed_z_m')
    if fz is None:
        return Tb
    dz = float(Tb[2, 3]) - fz
    print(f'[basket] marker z {Tb[2, 3]:.4f} m -> fixed {fz:.4f} m ({dz * 1000:+.1f} mm)', flush=True)
    if abs(dz) > cb.get('fixed_z_tol_m', 0.03):
        raise RuntimeError(f'basket marker fit z {Tb[2, 3]:.3f} m is {dz * 1000:+.0f} mm from the fixed {fz:.3f} m: '
                           'markers misread? check the marker GUI')
    Tb = np.array(Tb, float)
    Tb[2, 3] = fz
    return Tb


def region_pins(n, region, occupied=(), spacing=4):
    """(10/19 보스) candidate pins for "a plate in the front / back of the quadrant", in the order to try; pin 1 = farthest
    from the robot, n = nearest.  front: the robot half, nearest first (leaves room behind it).  back: farther than every
    plate already in this quadrant, nearest first; with none there, the far half.
    No candidate is closer than `spacing` pins to a plate already there (보스: red at 10 -> white at 6 or farther; with 2
    the white went in at 9, and real plates land ~1-2 pins off the aim)."""
    occupied = [int(o) for o in occupied]
    if region == 'front':
        cand = list(range(n, n // 2, -1))
    elif region == 'back':
        top = min(occupied) - spacing if occupied else n // 2
        cand = list(range(top, 0, -1))
    else:
        raise ValueError(f'region {region!r}: front or back')
    return [c for c in cand if all(abs(c - o) >= spacing for o in occupied)]


def landed_gap_error(Tbb, quadrant, pin, landed_pins):
    """(10/16) Gap error [x, y] in the basket frame (the sim's 'seen' gap error) from where earlier real inserts aimed
    at `pin` landed: mean landed gap centre - target gap centre (pin N = between tines N and N+1, fractions interpolate)."""
    from plate_plans import basket_gaps, pin_gap
    gaps = basket_gaps()[quadrant]
    n = len(gaps)
    bad = [v for v in [pin, *landed_pins] if not (np.isfinite(v) and 1 <= v <= n)]
    if bad or not len(landed_pins):                     # (10/18 Codex) np.interp would clamp out-of-range pins silently
        raise ValueError(f'{quadrant}: landed pins must be finite in 1..{n} (fractions allowed), got {bad or "none"}')
    x_of = lambda p_: float(np.interp(p_, np.arange(1, n + 1), [gaps[pin_gap(Tbb, quadrant, k)][0] for k in range(1, n + 1)]))
    return np.array([np.mean([x_of(l_) for l_ in landed_pins]) - x_of(pin), 0.0])


def basket_insert_policy_params(expert, quadrant, gap, H, Tbb, seen=None):
    """Use the same deterministic 16D/5D policy implementation as sim, with real-canonical H and unseen gap error
    (or `seen`: the gap error estimated from real inserts, for a policy trained seeing it)."""
    row = dict(name='real_pipeline', model_qd=quadrant, gap=int(gap))
    # `insert_policy_params` owns obs scaling, canonical_hand, tanh and INSERT_SCALE; DR is off and noseen is on.
    expert._CL_POL.update(det=True, dr=False, dr_full=False, noseen=seen is None, canon=True, cache={},
                          seen_override=None if seen is None else np.asarray(seen, float))
    result = expert.insert_policy_params(row, 0, np.asarray(H, dtype=float), np.asarray(Tbb, dtype=float))
    if (not np.isfinite(result['obs']).all() or not np.isfinite(result['act']).all()
            or not all(np.isfinite(result[k]) for k in ('dx', 'dy', 'lean', 'dz', 'yaw'))):
        raise RuntimeError('insert policy returned non-finite observation/action; refusing insertion replan')
    return result


def apply_basket_insert_policy(sim_kw, yaw_trim_deg, correction, tine_lean_deg):
    """Add the learned residual to the existing per-quadrant sim_q settings without changing any cap."""
    result = dict(sim_kw)
    result['x_shift_m'] = float(result['x_shift_m']) + float(correction['dx'])
    result['y_extra_m'] = float(result.get('y_extra_m', 0.0)) + float(correction['dy'])
    result['tilt_deg'] = float(result['tilt_deg']) + float(correction['lean']) * float(tine_lean_deg)
    result['centre_h'] = float(result['centre_h']) + float(correction['dz'])
    return result, float(yaw_trim_deg) + float(correction['yaw'])


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
    grip_force = None
    if args.plate_color == 'white':                         # (10/16 보스) flatter rim slips: deeper grasp, 2x force
        wp = C['white_plate']
        q['offset_m'] = [q['offset_m'][0] + wp['grasp_deeper_m'], q['offset_m'][1]]   # +radial = TCP further onto the plate
        grip_force = int(wp['grip_force'])
    insert_policy_expert = load_basket_insert_policy(args.insert_policy) if args.insert_policy else None
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
        if args.insert == 'quadrant' and (args.place != 'basket' or not args.quadrant):
            raise SystemExit('--insert quadrant needs --place basket (measured markers) and --quadrant Q1..Q4')
        if args.place in ('basket', 'teach'):                # markers first: the fixed D455 is FoundationPose's afterwards
            # (10/14 보스) teach: the basket pose is saved with every demonstration (BC per quadrant)
            from camera_tools import basket_fit_robust, basket_measure
            slot1 = json.loads((DATA / 'basket' / 'slot_01_in_basket.json').read_text())
            layout = json.loads(cpath(C['basket']['layout']).read_text())
            Tb, res, seen = basket_fit_robust(basket_measure(C), layout, C['basket'].get('max_fit_residual_m', 0.008))
            good = res <= C['basket'].get('max_fit_residual_m', 0.008) and len(seen) >= 3
            reuse = None if good or not args.basket_plates else basket_last_good(C['basket'])
            if reuse is not None:                           # (10/19 보스: '3 markers' on every 2nd plate) a plate standing in
                print(f'[basket] markers {seen} unreliable ({res * 1000:.1f} mm) with plates standing -> 저장된 바구니 위치 사용 '
                      f'({reuse[1] / 60:.0f} 분 전, 4마커)', flush=True)   # the basket hides part of a marker and misplaces it by
                Tb, (res, seen) = reuse[0], (0.0, ['last good'])   # cm; the basket has not moved since the 1st plate: reused
            Tb = fixed_basket_z(Tb, C['basket'])
            if good and len(seen) >= 4:
                save_last_good_basket(C['basket'], Tb, res, seen)
            T0 = np.array(slot1['T_base_basket_at_teach'])
            basket_delta = Tb @ np.linalg.inv(T0)
            basket_T = Tb
            mv = (Tb[:3, 3] - T0[:3, 3]) * 1000
            yaw = np.degrees(np.arctan2(basket_delta[1, 0], basket_delta[0, 0]))
            print(f'[basket] markers {seen}, fit {res * 1000:.1f} mm: moved x {mv[0]:+.0f} y {mv[1]:+.0f} mm, '
                  f'yaw {yaw:+.1f} deg since the teaching', flush=True)
            if reuse is None and (res > C['basket'].get('max_fit_residual_m', 0.008) or len(seen) < 3):
                raise RuntimeError(f'basket pose unreliable: {len(seen)} markers, residual {res * 1000:.1f} mm '
                                   '(empty basket: bash calibration/run.sh basket_check)')
        from sensors import FpStream
        fp = FpStream(udp['fp_pose'], st['n'], st['mm'], st['deg'], symmetric_z=True,   # round plate: yaw is noise
                      normalize=lambda T_: inv_T(Tbc) @ plate_upright(Tbc @ T_))       # (10/14) flips 179 deg <-> 1 deg
        vision_env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        prompt, mesh = q['prompt'], cpath(C['paths']['plate_obj'])
        if args.plate_color == 'white':                # (10/16 보스) white plate: same YCB 029 shape, white texture only
            prompt, mesh = C['white_plate']['prompt'], cpath(C['white_plate']['obj'])
            print(f'[plate] 흰 접시: YOLOE "{prompt}", FP mesh {mesh}', flush=True)
        perception = children.start('perception', [sys.executable, '-u', str(ROOT / 'real_time_project/main.py'),
                                                    '--detector', 'yoloe', '--width', '640', '--height', '480',
                                                    '--serial', args.serial, '--prompt', prompt, '--conf', str(args.conf),
                                                    '--mesh_file', str(mesh),
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
                viz(plate=plate_upright(Tbc @ T).tolist(), ok=bool(ok), area=None if q.get('library') else area)
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
        early, shown, policy_job, rep = None, plan, None, None
        if args.place == 'basket' and args.insert == 'quadrant':   # (10/13) any basket pose, IK carry + insert, no teaching
            legs0 = json.loads(Path(plan).read_text())['legs']
            from plate_plans import hand_plate, pin_gap, quadrant_plan, robot_quadrants
            H, tool = hand_plate(legs0['to_grasp']['waypoints_deg'][-1], p)
            mq = robot_quadrants(basket_T)[args.quadrant]
            if args.insert_landed_pins:                     # (10/18) check before any motion: a bad value must not stop a held plate
                landed_gap_error(basket_T, mq, args.pin, args.insert_landed_pins)
            rq = robot_quadrants(basket_T)                  # before motion: a bad pin raises here
            plates = [(rq[v.split(':')[0]], pin_gap(basket_T, rq[v.split(':')[0]], int(v.split(':')[1]))) if ':' in v
                      else (mq, pin_gap(basket_T, mq, int(v))) for v in (args.basket_plates or [])]
            qs = dict(H=H, tool=tool, yaw=args.yaw_trim if args.yaw_trim is not None else q.get('basket_quadrant_yaw_trim_deg', 0.0),
                      tilt=args.tilt_trim if args.tilt_trim is not None else q.get('basket_quadrant_tilt_trim_deg', 0.0))

            sim_kw = {}
            if args.sim_q:                                  # (10/15 보스 시연) Isaac per-quadrant lean / aim / wrist flip
                sys.path.insert(0, str(ROOT / 'rl'))
                import basket_env as be
                sq = C['basket_quadrant_sim'][mq]
                sim_kw = dict(tilt_deg=sq['lean'] * be.B['full']['tine_lean_deg'][mq], centre_h=sq['centre_h'] + q.get('basket_quadrant_sim_real_dh_m', 0.0),
                              x_shift_m=(sq['aim_m'] + q.get('basket_quadrant_model_dx_m', {}).get(mq, 0.0)) if args.x_shift is None else args.x_shift,
                              faces=(-1.0, 1.0) if sq['flip'] else (1.0, -1.0),
                              y_extra_m=q.get('basket_quadrant_sim_real_dy_m', {}).get(mq, 0.0))
                print(f'   sim 설정 {mq}: {sq} -> 기울기 {sim_kw["tilt_deg"]:+.2f} deg, 겨냥 {sim_kw["x_shift_m"]:+.4f} m', flush=True)

            def qplan(start_deg):                           # the insert plan for the current hand model and trims
                kw = dict(centre_h=q.get('basket_quadrant_centre_h_m', 0.18), tilt_deg=-qs['tilt'], x_shift_m=args.x_shift)
                kw.update(sim_kw)
                if qs.get('faces'):                         # (10/19) region mode: keep the face the pin was chosen with
                    kw['faces'] = qs['faces']
                if qs.get('phi_first') is not None:         # (10/19 보스) the replan after the lift: the chosen phi first
                    kw['phi_first'] = qs['phi_first']
                return quadrant_plan(basket_T, mq, pin_gap(basket_T, mq, args.pin), qs['H'], qs['tool'], start_deg, Tbc, q,
                                     yaw_trim_deg=qs['yaw'], y_shift_m=args.y_shift, grasp_deg=legs0['to_grasp']['waypoints_deg'][-1],
                                     plates=plates, grasp_low_z=float(p[2]), **kw)   # (10/19) 15 cm from the plate on the stand
            if args.region:                                 # (10/19 보스) any pin of the front / back half that has a plan
                from plate_plans import basket_gaps
                same_q = [int(v.split(':')[-1]) for v in (args.basket_plates or []) if ':' not in v or v.split(':')[0] == args.quadrant]
                cands = region_pins(len(basket_gaps()[mq]), args.region, same_q, int(q.get('basket_region_min_spacing', 4)))
                print(f'   {args.quadrant} {args.region}: 후보 핀 {cands} (이미 있는 접시 {same_q})', flush=True)
                early, rep, why = None, None, {}
                order = sim_kw.get('faces', (1.0, -1.0))   # (10/19 보스: the wrist spun ~170 deg on Q3 pin 12) every pin with the
                for faces_ in ((order[0],), tuple(order)):  # usual plate face first, the flipped wrist only if none has a plan
                    qs['faces'] = faces_
                    for pin_ in cands:
                        args.pin = pin_
                        early, rep = qplan(legs0['lift']['waypoints_deg'][-1])
                        if early is not None:
                            break
                        why[(pin_, len(faces_))] = sorted({w.split(':')[0] if ':' in w else w.split(' ')[0] for _, w in rep['tried']})
                        print(f'   핀 {pin_} ({"평소 면" if len(faces_) == 1 else "손목 뒤집기 포함"}): 경로 없음 '
                              f'{why[(pin_, len(faces_))]}', flush=True)
                    if early is not None:
                        break
                qs['faces'] = (float(rep['face']),) if early is not None else None
                if early is None:
                    raise RuntimeError(f'{args.quadrant} {args.region}: no pin with an insert plan ({why})')
                if args.chosen_pin_out:
                    Path(args.chosen_pin_out).write_text(f'{args.quadrant}:{args.pin}\n')
            else:
                early, rep = qplan(legs0['lift']['waypoints_deg'][-1])
            if early is None:
                raise RuntimeError(f'{args.quadrant} pin {args.pin}: no insert plan from this grasp ({rep["tried"][:4]})')
            print(f'   {args.quadrant} (바구니 모델 {mq}) 핀 {args.pin}: 넣기 계획 OK, 운반 {rep["carry"]}, '
                  f'충돌 여유/기준 {rep["clearance_ratio"]}', flush=True)
            both = json.loads(Path(plan).read_text())
            both['legs'].update(json.loads(early.read_text())['legs'])
            shown = out / 'preview_plan.json'
            shown.write_text(json.dumps(dict(both, preview_only=True, basket_T=np.asarray(basket_T).tolist(),
                                             plate_in_tcp=np.asarray(H).tolist())))
        elif args.place in ('golden', 'basket'):          # (10/12) whole path before moving: reach check + RViz preview
            legs0 = json.loads(Path(plan).read_text())['legs']
            early = golden_slot_plan(q['golden_teach'], legs0['to_grasp']['waypoints_deg'][-1], legs0['lift']['waypoints_deg'][-1],
                                     Tbc, q, basket_delta=basket_delta, plate_xy=p[:2])
            both = json.loads(Path(plan).read_text())
            both['legs'].update(json.loads(early.read_text())['legs'])
            shown = out / 'preview_plan.json'
            shown.write_text(json.dumps(dict(both, preview_only=True)))
            print(f'   바구니 경로까지 계획 완료 (RViz 미리보기: 잡기 + 바구니 전체): {early}', flush=True)
            if args.insert == 'policy':                 # (10/13) learned insertion: sim rollout in parallel with the motion
                Tbb = basket_T if basket_T is not None else np.array(json.loads(
                    (DATA / 'basket' / 'slot_01_in_basket.json').read_text())['T_base_basket_at_teach'])
                if basket_T is None:
                    Tbb = Tbb.copy()
                    Tbb[:3, 3] += q.get('basket_shift_m', [0.0, 0.0, 0.0])
                req = dict(T_base_basket=np.asarray(Tbb).tolist(), grasp_deg=legs0['to_grasp']['waypoints_deg'][-1],
                           plate_root_fp=[float(v) for v in p], slot_shift_pitches=q.get('basket_slot_shift_pitches', 2),
                           above_deg=json.loads(early.read_text())['legs']['to_above']['waypoints_deg'][-1],
                           out=str(out / 'realplan.json'))
                (out / 'realplan_req.json').write_text(json.dumps(req, indent=1))
                policy_job = subprocess.Popen([q['isaaclab'].replace('isaaclab.sh', '_isaac_sim/python.sh'),
                                               str(ROOT / 'rl/basket/expert.py'), 'realplan', '--headless', '--per_run', '8',
                                               '--req', str(out / 'realplan_req.json'), '--policy', str(cpath(q['basket_policy_bc'])),
                                               '--residual', str(cpath(q['basket_policy_rl']))],
                                              cwd=ROOT / 'rl', env=isaac_env(), stdin=subprocess.DEVNULL,
                                              stdout=(out / 'realplan.log').open('wb'), stderr=subprocess.STDOUT)
                print(f'   학습 정책 넣기 경로를 sim에서 계산 중 (pid {policy_job.pid}, {out / "realplan.log"})', flush=True)
                from plate_plans import basket_range_note
                basket_range_note(Tbb, q)
        viz(plan=str(shown))
        print(f'   plan: {plan}  (30 분 유효)', flush=True)
        if not args.execute:
            print('[5/5] 계획만 만들었습니다. --execute 로 실제 이동.', flush=True)
            return
        vel = q.get('vel_deg_s', 2.0)

        def leg(name):
            v = vel
            print(f'\n=== {name}: one smooth spline at {v} deg/s ===', flush=True)
            rec.event('leg_start', leg=name, vel_deg_s=v, smooth=True)
            children.run_mover(mover_command(args, plan, name, 'leg', opening_done=True, smooth=True, vel=v), env)
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
                gripper(['close' if ans == 'c' else 'open', '--yes']
                        + (['--force', str(grip_force)] if ans == 'c' and grip_force else []), env)
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
            viz(plate=plate_upright(Tbc @ fp.hist[-1]).tolist(), ok=True)
            return dict(plate_root=[round(float(v), 5) for v in root], yaw_deg=round(fyaw, 2), tilt_deg=round(ftilt, 2),
                        T=np.round(plate_upright(Tbc @ fp.hist[-1]), 5).tolist())     # (10/13) full pose: plate normal in the hand afterwards
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
        merge = args.auto and args.insert in ('policy', 'quadrant') and q.get('auto_merged_legs', False) and early is not None   # (10/13 보스)
        lift_alone = not merge or args.insert == 'quadrant'     # quadrant: stop after the lift to see the plate in the hand
        if lift_alone:
            leg('lift')
            rec.event('lifted')
            time.sleep(q.get('hold_s', 3.0))
        lift_end = json.loads(Path(plan).read_text())['legs']['lift']['waypoints_deg'][-1]
        if args.insert == 'quadrant':                       # (10/13 실기) the plate sags in the fingers: replan from the FP plate now
            fp.poll()
            if fp.hist and rec.last:
                Tn = plate_upright(Tbc @ fp.hist[-1])
                root, _, tilt_now = plate_root_in_base(Tn)
                from plate_plans import hand_plate, pin_gap, quadrant_plan, robot_quadrants
                n_now = Tn[:3, :3] @ plate_upright(Tbc @ T)[:3, :3].T @ [0, 0, 1.0]   # change since the flat plate: FP's ~2-4 deg bias cancels
                H2, tool2 = hand_plate(rec.last['posj'], root, n_now)
                if insert_policy_expert is not None:
                    sim_kw_before_policy, yaw_before_policy = dict(sim_kw), qs['yaw']
                    seen = (landed_gap_error(basket_T, mq, args.pin, args.insert_landed_pins)
                            if args.insert_landed_pins else None)
                    if seen is not None:
                        print(f'   실기 착지 기록 {args.insert_landed_pins} -> 칸 오차 x {seen[0]*1000:+.1f} mm (정책 입력)', flush=True)
                    correction = basket_insert_policy_params(insert_policy_expert, mq,
                                                               pin_gap(basket_T, mq, args.pin), H2, basket_T, seen=seen)
                    sim_kw, qs['yaw'] = apply_basket_insert_policy(
                        sim_kw, qs['yaw'], correction, be.B['full']['tine_lean_deg'][mq])
                    rec.event('insert_policy_correction', checkpoint=str(args.insert_policy.resolve()),
                              observation=np.asarray(correction['obs']).astype(float).tolist(),
                              raw_mean=np.asarray(correction['act']).astype(float).tolist(),
                              dx_m=correction['dx'], dy_m=correction['dy'], lean=correction['lean'],
                              dz_m=correction['dz'], yaw_deg=correction['yaw'],
                              applied=sim_kw)
                    print(f'   삽입 정책 평균 보정: dx {correction["dx"]:+.4f} m, dy {correction["dy"]:+.4f} m, '
                          f'lean {correction["lean"]:+.3f}, dz {correction["dz"]:+.4f} m, yaw {correction["yaw"]:+.2f} deg',
                          flush=True)
                print(f'   들어 올린 뒤 접시 (FP): 기울기 {tilt_now:.1f} deg (잡기 전 {tilt:.1f}) -> 손 안 접시 자세로 넣기 경로 다시 계산',
                      flush=True)
                H_old = qs['H'], qs['tool']
                qs['H'], qs['tool'] = H2, tool2
                if early is not None and rep:               # (10/19 보스 A) same face and phi first: ~1 s instead of ~6 s
                    qs['faces'] = tuple(dict.fromkeys([float(rep['face']), *(qs.get('faces') or sim_kw.get('faces', (1.0, -1.0)))]))
                    qs['phi_first'] = rep['phi_deg']
                t_re = time.monotonic()
                again, rep2 = qplan(rec.last['posj'])
                print(f'   다시 계산 {time.monotonic() - t_re:.1f} s', flush=True)
                if again is None:
                    qs['H'], qs['tool'] = H_old
                    if insert_policy_expert is not None:
                        sim_kw, qs['yaw'] = sim_kw_before_policy, yaw_before_policy
                        raise RuntimeError(f'policy-corrected insert plan refused ({rep2["tried"][:3]}); '
                                           'will not fall back to the uncorrected pre-lift plan')
                if again is not None:
                    early = again
                    rec.event('replan_after_lift', plan=str(again), tilt_deg=tilt_now, plate_root=[float(v) for v in root],
                              normal=np.asarray(n_now).tolist())
                    print(f'   다시 계산 OK: {again} (충돌 여유/기준 {rep2["clearance_ratio"]})', flush=True)
                else:
                    print(f'   다시 계산 실패 ({rep2["tried"][:3]}): 처음 계획으로 진행', flush=True)
            else:
                if insert_policy_expert is not None:
                    raise RuntimeError('--insert_policy needs a fresh post-lift FoundationPose plate pose; refusing the uncorrected fallback')
                print('   FP 접시 없음: 처음 계획으로 진행', flush=True)
        if args.place == 'teach':
            k = args.slot or next_slot()
            teach_slot(k, rec, choose_gripper, lift_end, meta=dict(
                run=str(out), quadrant=args.quadrant, plate_root_fp=[float(v) for v in p],
                T_base_basket=None if basket_T is None else np.asarray(basket_T).tolist()))
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

            def sleg(name, tool_force_limit_n=None):
                # (10/13 보스) only the plate's way down is slow; every other leg runs at the one speed vel_deg_s
                v = {'descend_pre': q.get('basket_pre_vel_deg_s', vel), 'to_policy_start': q.get('basket_pre_vel_deg_s', vel),
                     'descend': q.get('basket_insert_vel_deg_s', vel), 'shift': q.get('basket_insert_vel_deg_s', vel),
                     'policy_descend': q.get('basket_policy_vel_deg_s', 1.0)}.get(name, vel)
                print(f'\n=== slot {k} {name}: one smooth spline at {v} deg/s ===', flush=True)
                rec.event('leg_start', leg=f'slot{k}_{name}', vel_deg_s=v, smooth=True)
                children.run_mover(mover_command(args, cur['plan'], name, 'leg', opening_done=True, smooth=True, vel=v,
                                                 tool_force_limit_n=tool_force_limit_n), env)
                rec.event('leg_end', leg=f'slot{k}_{name}')

            def mleg(parts, name, v, tool_force_limit_n=None): # merged legs: one spline, no stop between them
                mp = merged_plan(parts, name)
                if mp is None:                                  # too many waypoints: the legs one after another
                    print(f'   [merged] {name}: 묶을 수 없음 -> 구간별 실행', flush=True)
                    for pp, lg in parts:
                        cur['plan'] = pp
                        sleg(lg, tool_force_limit_n if lg == 'descend' else None)
                    return
                print(f'\n=== slot {k} {name}: ONE spline ({len(parts)} legs) at {v} deg/s ===', flush=True)
                rec.event('leg_start', leg=f'slot{k}_{name}', vel_deg_s=v, smooth=True, merged=[lg for _, lg in parts])
                children.run_mover(mover_command(args, mp, name, 'leg', opening_done=True, smooth=True, vel=v,
                                                 tool_force_limit_n=tool_force_limit_n), env)
                rec.event('leg_end', leg=f'slot{k}_{name}')
            if 'to_above' in json.loads(Path(splan).read_text())['legs']:     # golden / basket: four stages
                def look(tag):                                      # read-only wrist photo + model pins (10/12)
                    try:                                            # child process: OpenCV stays out of this one
                        t_ask = time.time()                         # joints are not streamed during a move: wait for a fresh one
                        while (rec.last or {}).get('t', 0) < t_ask and time.time() - t_ask < 5:
                            time.sleep(0.05)
                        if (rec.last or {}).get('t', 0) < t_ask:    # (10/19 Codex) no fresh joints: no photo with a stale pose
                            raise RuntimeError('no fresh joint state within 5 s')
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
                        share = Path('/dev/shm/fixed_latest.npz')         # (10/14) fixed RGB-D at the same stop: held-plate angle (L)
                        t_f = time.time()
                        while share.exists() and float(np.load(share)['t']) < t_ask and time.time() - t_f < 3:
                            time.sleep(0.1)
                        if share.exists() and float(np.load(share)['t']) >= t_ask:
                            import shutil
                            shutil.copy(share, out / f'fixed_{tag}.npz')
                            (out / f'fixed_{tag}.json').write_text(json.dumps(dict(
                                T_base_cam=np.asarray(Tbc).tolist(), T_base_basket=Tbb.tolist(), posj=rec.last['posj'],
                                plan=str(cur['plan']))))
                            print(f'   고정 카메라 RGB-D: {out / f"fixed_{tag}.npz"}', flush=True)
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
                if merge:
                    mleg([(splan, 'to_above')] if lift_alone else [(plan, 'lift'), (splan, 'to_above')], 'carry', vel)
                else:
                    sleg('to_above')
                    look('above')
                if args.stop_at_above:
                    print('[1/4] 바구니 위에서 멈춤 (--stop_at_above). 접시는 잡은 채 그대로입니다.', flush=True)
                    rec.event('done', place=args.place, slot=k, stopped='above')
                    return
                while args.insert == 'quadrant' and not args.auto:   # (10/14 보스) fix the plate angle right here, no trip home
                    a_ = input(f'\n[1/4] 바구니 위 (회전 보정 {qs["yaw"]:+.1f}, 기울기 보정 {qs["tilt"]:+.1f} deg). d = 넣기, h = 홈 복귀, '
                               f'숫자 = 접시를 그만큼 더 돌림(deg, + 반시계, 위에서 봄), t숫자 = 앞뒤 기울기 더(deg, + 위가 먼 쪽) > ').strip().lower()
                    if a_ in ('d', 'h'):
                        break
                    try:
                        key, val = ('tilt', float(a_[1:])) if a_.startswith('t') else ('yaw', float(a_))
                    except ValueError:
                        continue
                    if abs(val) > 30:
                        print('   한 번에 30 deg까지만', flush=True)
                        continue
                    qs[key] += val
                    new_plan, rep3 = qplan(rec.last['posj'])
                    if new_plan is None:
                        qs[key] -= val
                        print(f'   그 각도로는 경로 없음 ({rep3["tried"][:3]}): 그대로', flush=True)
                        continue
                    splan = new_plan
                    cur['plan'] = splan
                    rec.event('above_trim', yaw_trim=qs['yaw'], tilt_trim=qs['tilt'], plan=str(splan))
                    print(f'   다시 계산: 회전 보정 {qs["yaw"]:+.1f}, 기울기 보정 {qs["tilt"]:+.1f} -> 바구니 위 자세로 이동', flush=True)
                    sleg('to_above')
                    look('above')
                else:
                    a_ = ask('\n[1/4] 바구니 위. d = 2단계 하강 시작, h = 하강 없이 홈 복귀 > ', ('d', 'h'))
                if a_ == 'h':
                    sleg('abort_home')
                    rec.event('done', place=args.place, slot=k, stopped='above_home')
                    return
                if args.insert == 'quadrant':                  # straight down at the insert speed, release, straight up + home
                    if merge:
                        look('above')
                    mleg([(splan, 'descend')], 'insert', q.get('basket_quadrant_vel_deg_s', q.get('basket_policy_vel_deg_s', 1.0)),
                         tool_force_limit_n=args.insert_force_limit_n)
                    choose_gripper(f'slot{k}_release')
                    if args.photo_after_release:            # (10/19 보스) opt-in: empty hand back at the above pose -> wrist photo of
                        mleg([(splan, 'ascend')], 'retreat_up', vel)   # the target quadrant without the held plate (where it landed)
                        look('after_release')
                        mleg([(splan, 'to_home')], 'retreat_home', vel)
                    else:
                        mleg([(splan, 'ascend'), (splan, 'to_home')], 'retreat', vel)
                    rec.event('done', place=args.place, quadrant=args.quadrant, pin=args.pin, insert='quadrant', merged=merge)
                    print(f'[5/5] {args.quadrant} 핀 {args.pin} 넣기 완료.', flush=True)
                    return
                if policy_job is not None:
                    ppath = policy_plan(policy_job, out, Tbc, q)
                    if ppath is not None:
                        cur['plan'] = ppath
                        rec.event('policy_plan', plan=str(ppath))
                        if merge:
                            mleg([(ppath, 'to_policy_start'), (ppath, 'policy_descend')], 'insert', q.get('basket_policy_vel_deg_s', 1.0))
                            choose_gripper(f'slot{k}_release')
                            mleg([(ppath, 'policy_ascend'), (ppath, 'to_home')], 'retreat', vel)
                            rec.event('done', place=args.place, slot=k, insert='policy', merged=True)
                            print('[5/5] 학습 정책으로 넣기 완료 (spline 4개).', flush=True)
                            return
                        sleg('to_policy_start')
                        look('pre')
                        if ask('\n[2/4] 학습 정책 시작 자세. d = 넣기(정책 경로), u = 다시 올려 홈 복귀 > ', ('d', 'u')) == 'u':
                            sleg('policy_abort')
                            rec.event('done', place=args.place, slot=k, stopped='policy_start')
                            return
                        sleg('policy_descend')
                        choose_gripper(f'slot{k}_release')
                        sleg('policy_ascend')
                        sleg('to_home')
                        rec.event('done', place=args.place, slot=k, insert='policy')
                        print('[5/5] 학습 정책으로 넣기 완료.', flush=True)
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
            meta = dict(run=str(out), quadrant=args.quadrant)
            try:                                               # (10/14 보스) BC demos: the basket pose with every teaching
                from camera_tools import basket_fit_robust, basket_measure
                layout = json.loads(cpath(C['basket']['layout']).read_text())
                Tb, res, seen = basket_fit_robust(basket_measure(C), layout, C['basket'].get('max_fit_residual_m', 0.010))
                meta.update(T_base_basket=np.asarray(Tb).tolist(), basket_fit_mm=round(res * 1000, 2), basket_markers=list(map(int, seen)))
                print(f'[basket] markers {seen}, fit {res * 1000:.1f} mm, centre ({Tb[0, 3]:.3f}, {Tb[1, 3]:+.3f}) '
                      f'yaw {np.degrees(np.arctan2(Tb[1, 0], Tb[0, 0])):+.1f} deg', flush=True)
            except Exception as e:                             # teaching still goes on; the pose is missing from this demo
                print(f'[basket] 측정 실패 ({e}): 이 시연에는 바구니 자세 없이 저장', flush=True)
            choose_gripper(f'teach{k}_start (열기 o 권장)')
            teach_full(k, lambda: rec.last, choose_gripper, rec.event, meta=meta)
            if not args.slot:                                 # (10/14) one file per demonstration
                set_next_full(k + 1)
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
            v = vel
            print(f'\n=== {name}: one smooth spline at {v} deg/s ===', flush=True)
            rec.event('leg_start', leg=name, vel_deg_s=v, smooth=True)
            children.run_mover(mover_command(args, plan, name, 'leg', opening_done=True, smooth=True, vel=v), env)
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
    p.add_argument('--insert', choices=('taught', 'policy', 'quadrant'), default='taught', help='golden / basket: taught descent (default), the '
                   'learned policy (BC + residual RL, rl/basket) rolled out in sim on the measured basket / grasp, or quadrant = '
                   'IK carry + straight-down insert into --quadrant / --pin of the measured basket (10/13, no teaching)')
    p.add_argument('--quadrant', choices=('Q1', 'Q2', 'Q3', 'Q4'), default=None,
                   help='--insert quadrant: robot view, robot at the bottom: Q1 far-right, Q2 far-left, Q3 near-left, Q4 near-right')
    p.add_argument('--yaw_trim', type=float, default=None, help='--insert quadrant: turn the plate face about the vertical at '
                   'the insert [deg], + = counter-clockwise seen from above (default config basket_quadrant_yaw_trim_deg)')
    p.add_argument('--tilt_trim', type=float, default=None, help='--insert quadrant: lean the plate along the pin rows at the '
                   'insert [deg], + = plate top toward the far side (default config basket_quadrant_tilt_trim_deg)')
    p.add_argument('--y_shift', type=float, default=None, help='--insert quadrant: plate centre along the pin rows\' normal '
                   '(basket y, + = left seen from the robot) [m]; default: automatic, keeps the 26.8 cm rim off the side wall')
    p.add_argument('--x_shift', type=float, default=None, help='--insert quadrant: plate along the pin rows (basket x, + = away '
                   'from the robot) [m]; default config basket_quadrant_x_trim_m')
    p.add_argument('--sim_q', action='store_true', help='--insert quadrant: use config basket_quadrant_sim (Isaac per-quadrant '
                   'lean, aim, wrist flip; 10/15) instead of the real trims; --x_shift still overrides the aim')
    p.add_argument('--photo_after_release', action='store_true',
                   help='(10/19 보스) --insert quadrant: after the release, stop at the above pose with the empty hand and take a '
                        'wrist photo (wrist_after_release.*: target quadrant + the plate just inserted, live joints), then go home')
    p.add_argument('--basket_plates', nargs='+', default=None, metavar='[Q:]PIN',
                   help='(10/18 보스) --insert quadrant: plates already in the basket, PIN (this --quadrant) or Q4:10 (any quadrant); '
                        'the carry passes over them (collision boxes), the straight-down insert beside them')
    p.add_argument('--insert_landed_pins', type=float, nargs='+', default=None,
                   help='(10/16) with --insert_policy (a policy trained seeing the gap error): pins where earlier real inserts '
                        'aimed at this --pin landed (e.g. 3.5 = between 3 and 4); their mean error is the policy\'s gap input')
    p.add_argument('--insert_policy', '--insert-policy', type=Path, default=None,
                   help='OPT-IN: apply the deterministic mean of a basket insert PPO checkpoint after lift, using canonical hand pose and unseen gap error; requires --task plate --place basket --insert quadrant --sim_q --execute')
    p.add_argument('--region', choices=('front', 'back'), default=None,
                   help='(10/19 보스) --insert quadrant: instead of --pin, the first pin of this half of the quadrant that has a plan '
                        '(front = robot half, back = behind the plates already there); see region_pins')
    p.add_argument('--chosen_pin_out', type=Path, default=None, help='with --region: write the chosen "Q:pin" here (demo4)')
    p.add_argument('--pin', type=int, default=5, help='--insert quadrant: pin-pair number in the quadrant, 1 = farthest from '
                   'the robot (wrist image top -> bottom); the plate goes into the gap just robot-side of that pair')
    p.add_argument('--insert-force-limit-n', type=float, default=None,
                   help='OPT-IN real --place basket --insert quadrant only: stop the insert spline if base-frame tool-force norm exceeds N; default off')
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
    p.add_argument('--plate_color', choices=('red', 'white'), default='red',
                   help='(10/16) plate: white = the white plate (YOLOE prompt + white-textured 029 mesh, same shape); default red')
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
    if a.insert_force_limit_n is not None:
        if not np.isfinite(a.insert_force_limit_n) or a.insert_force_limit_n <= 0:
            p.error('--insert-force-limit-n must be finite and > 0')
        if not (a.execute and a.task == 'plate' and a.place == 'basket' and a.insert == 'quadrant' and not a.offline_test):
            p.error('--insert-force-limit-n requires --task plate --execute --place basket --insert quadrant; default remains off')
    if a.plate_color == 'white' and a.no_library:
        p.error('--plate_color white uses the library grasp (its +1 cm is radial there); not with --no_library')
    if a.insert_landed_pins and not all(np.isfinite(v) and v >= 1 for v in a.insert_landed_pins):
        p.error('--insert_landed_pins must be finite pin numbers >= 1 (fractions like 7.5 allowed)')
    for v in a.basket_plates or []:
        q_, _, pin_ = v.rpartition(':')
        if not pin_.isdigit() or int(pin_) < 1 or q_ not in ('', 'Q1', 'Q2', 'Q3', 'Q4'):
            p.error(f'--basket_plates {v}: PIN >= 1 or Q1..Q4:PIN')
        if not a.region and int(pin_) == a.pin and q_ in ('', a.quadrant):
            p.error('--basket_plates: the target --pin already holds a plate')
    if a.region and a.insert_landed_pins:
        p.error('--insert_landed_pins is for a fixed --pin, not --region')
    if a.insert_landed_pins and a.insert_policy is None:
        p.error('--insert_landed_pins needs --insert_policy (a policy trained seeing the gap error)')
    if a.insert_policy is not None:
        if not (a.task == 'plate' and a.place == 'basket' and a.insert == 'quadrant' and a.quadrant
                and a.sim_q and a.execute and not a.offline_test):
            p.error('--insert_policy requires --task plate --place basket --insert quadrant --quadrant Q1..Q4 --sim_q --execute')
        if not a.insert_policy.is_file():
            p.error(f'--insert_policy checkpoint does not exist: {a.insert_policy}')
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
