"""One-terminal cube pipeline. Default: plan + ROS publication, no motion.

--execute hands the frozen plan to the existing mover; every waypoint still
requires a human's go. --offline-test uses synthetic poses, no ROS or camera.
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

import numpy as np

import real_pose_dryrun as dry
import ros2_pendant_mover as mover
from calib_utils import inv_T, load_json, rot_angle_deg, save_json
from kinematics import DEFAULT_URDF, Kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='human-supervised real motion, one go per waypoint')
    p.add_argument('--offline-test', action='store_true', help='synthetic pose -> plan -> mover validation; no hardware')
    p.add_argument('--serial', default='338122300585',
                   help='FIXED D455 serial (the wrist D455 is 338122303684; an empty serial would pick either one)')
    p.add_argument('--t_base_cam', type=Path, help='default: existing latest-session selection, including zfix')
    p.add_argument('--cell_json', type=Path, default=dry.DATA / 'real_cell_measured.json')
    p.add_argument('--tool_length_m', type=float, default=0.0)
    p.add_argument('--tool_radius_m', type=float, default=0.05)
    p.add_argument('--pregrasp_height_m', type=float, default=0.10)
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--prompt', default="rubik's cube")
    p.add_argument('--timeout', type=float, default=180.0, help='seconds allowed to obtain a stable pose/plan')
    p.add_argument('--robot_id', default='dsr01')
    p.add_argument('--controller_ns', default='dsr_controller2')
    p.add_argument('--ros_setup', type=Path, default=Path('/opt/ros/jazzy/setup.bash'))
    p.add_argument('--doosan_setup', type=Path, default=ROOT.parent / 'doosan_ws/install/setup.bash')
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


def ros_environment(args):
    """Source ROS for system-Python children without leaking conda's Python ABI."""
    for path in (args.ros_setup, args.doosan_setup):
        if not path.is_file():
            raise RuntimeError(f'ROS setup file missing: {path}')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH',
                'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH'):
        env.pop(key, None)
    env['PATH'] = '/usr/bin:/bin'
    res = subprocess.run(
        ['/bin/bash', '-c', 'source "$1" >&2 && source "$2" >&2 && exec /usr/bin/env -0',
         'ros-env', str(args.ros_setup), str(args.doosan_setup)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
    return dict(item.decode().split('=', 1) for item in res.stdout.split(b'\0') if item)


class Children:
    """Only terminate subprocesses owned by this invocation, including on failures."""
    def __init__(self, out):
        self.out, self.items = out, []
        self.active_mover = None

    def start(self, name, command, env=None):
        log = self.out / f'{name}.log'
        with log.open('wb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
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
        # Same foreground process group: Ctrl+C reaches the mover's stop handler.
        self.active_mover = subprocess.Popen(command, cwd=ROOT, env=env)
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


class Receiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Exclusive bind: a leftover planner/bridge must not steal this session's poses.
            self.sock.bind(('127.0.0.1', port))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise

    def packets(self):
        while True:
            try:
                raw = self.sock.recv(65535)
            except BlockingIOError:
                return
            try:
                msg = json.loads(raw)
                if isinstance(msg, dict):
                    yield msg
            except (ValueError, UnicodeDecodeError):
                continue


def fresh_joints(msg, now=None):
    now = time.time() if now is None else now
    try:
        q = np.asarray(msg['posj'], dtype=float)
        age = now - float(msg['t'])
        return q.shape == (6,) and np.isfinite(q).all() and 0 <= age < 1.0
    except (KeyError, TypeError, ValueError):
        return False


class StablePose:
    def __init__(self, T_base_cam, args):
        self.T_base_cam, self.args = T_base_cam, args
        self.history = collections.deque(maxlen=args.stable_n)
        self.last_stamp = 0

    def add(self, msg, now=None):
        now = time.time() if now is None else now
        try:
            timestamp = msg['timestamp']
            stamp = float(timestamp['sec']) + float(timestamp['nanosec']) / 1e9
            T = np.asarray(msg['pose'], dtype=float)
            if (T.shape != (4, 4) or not np.isfinite(T).all()
                    or not np.allclose(T[3], [0, 0, 0, 1])
                    or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-3)):
                self.history.clear()
                return None
            if not np.isfinite(stamp) or not 0 <= now - stamp < 2 or stamp <= self.last_stamp:
                self.history.clear()
                return None
        except (KeyError, TypeError, ValueError):
            self.history.clear()
            return None
        if self.last_stamp and stamp - self.last_stamp > 1.0:
            self.history.clear()
        self.last_stamp = stamp
        self.history.append(self.T_base_cam @ T)
        if len(self.history) < self.args.stable_n:
            return None
        ref = self.history[-1]
        if (max(np.linalg.norm(h[:3, 3] - ref[:3, 3]) * 1000 for h in self.history) > self.args.stable_mm
                or max(rot_angle_deg(h[:3, :3].T @ ref[:3, :3]) for h in self.history) > self.args.stable_deg):
            return None
        result = ref.copy()
        result[:3, 3] = np.median([h[:3, 3] for h in self.history], axis=0)
        return result


def make_plan(kin, checker, T, q_deg, args, t_path):
    try:
        res = dry.plan_dryrun(kin, checker, T, np.radians(q_deg), args)
    except RuntimeError as exc:
        raise RuntimeError(f'path planning refused: {exc}') from exc
    res.update(t_base_cam_file=str(t_path), cell=str(args.cell_json),
               created=time.strftime('%Y-%m-%d %H:%M:%S'),
               joint_source='live_ros2', pipeline='cube_pipeline')
    if res['accepted']:
        problems = []
        for leg in ('approach', 'return'):
            a = mover.parse_args(['--plan', 'unused', '--leg', leg])
            problems.extend(mover.validate(a, res)[1])
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
    dry.report(res)
    print(f'Plan: {path}', flush=True)
    return path


def mover_command(args, path, leg):
    cmd = ['/usr/bin/python3', '-u', str(HERE / 'ros2_pendant_mover.py'),
           '--plan', str(path), '--leg', leg, '--robot_id', args.robot_id,
           '--controller_ns', args.controller_ns]
    if args.execute:
        cmd.append('--execute')
    return cmd


def wait_choice(children):
    print('\n도착 완료. 복귀하려면 return, 현재 위치에서 종료하려면 quit 입력: ', end='', flush=True)
    while True:
        children.check()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            return sys.stdin.readline().strip() == 'return'


def run(args):
    t_path = (args.t_base_cam or dry.latest_t_base_cam()).resolve()
    Tbc = np.asarray(load_json(t_path)['T_base_cam'], dtype=float)
    if (Tbc.shape != (4, 4) or not np.isfinite(Tbc).all()
            or not np.allclose(Tbc[3], [0, 0, 0, 1])
            or not np.allclose(Tbc[:3, :3].T @ Tbc[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(Tbc[:3, :3]), 1, atol=1e-3)):
        raise RuntimeError(f'invalid T_base_cam: {t_path}')
    pargs = dry.parse_args([])
    for name in ('cell_json', 'tool_length_m', 'tool_radius_m', 'pregrasp_height_m'):
        setattr(pargs, name, getattr(args, name))
    kin = Kinematics(DEFAULT_URDF)
    checker, _ = dry.build_checker(kin, pargs, Tbc)
    parent = dry.DATA / ('pipeline_offline' if args.offline_test else 'real_pipeline')
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_'), dir=parent))
    print(f'Mode: {"OFFLINE" if args.offline_test else "SUPERVISED" if args.execute else "PLAN ONLY"}\n'
          f'T_base_cam: {t_path}\nCell: {args.cell_json}\nLogs: {out}', flush=True)
    if 'zfix' in t_path.name:
        print('현재 변환: zfix (마커 데이터 위치 잔차 평균 23.3 mm; 큐브 GT 미검증).', flush=True)
    stable = StablePose(Tbc, pargs)
    if args.offline_test:
        Tbase = np.eye(4)
        Tbase[:3, 3] = [1.04, 0.05, 0.0285]
        Tcam = inv_T(Tbc) @ Tbase
        now = time.time()
        for i in range(pargs.stable_n):
            stamp = now - 0.2 + i * 0.01
            T = stable.add(dict(pose=Tcam.tolist(), timestamp=dict(sec=int(stamp),
                           nanosec=int((stamp % 1) * 1e9))), now=now)
        if T is None:
            raise RuntimeError('offline pose stability check failed')
        res = make_plan(kin, checker, T, np.zeros(6), pargs, t_path)
        res['joint_source'] = 'synthetic_offline'
        res['offline_test'] = True
        # An offline fixture must never qualify as a plan for real motion.
        res['execution_allowed'] = False
        save_plan(out, res)
        if not res['accepted']:
            raise RuntimeError('offline plan refused: ' + '; '.join(res['reasons']))
        print('OFFLINE OK: transform -> stability -> IK/collision -> approach/return validation. No ROS/camera.', flush=True)
        return

    children, receivers = Children(out), []
    try:
        env = ros_environment(args)
        for port in (5005, 5006):
            receivers.append(Receiver(port))
        poses, joints = receivers
        children.start('robot_state', ['/usr/bin/python3', '-u', str(HERE / 'ros2_flange_udp.py'),
                       '--robot_id', args.robot_id, '--controller_ns', args.controller_ns], env)
        ready, ack = out / 'publisher.ready', out / 'publisher.ack'
        children.start('ros_publisher', ['/usr/bin/python3', '-u', str(HERE / 'ros2_dryrun_publisher.py'),
                       '--ready-file', str(ready), '--ack-file', str(ack)], env)
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
        # Conda inference does not need ROS's system-Python packages.
        vision_env.pop('PYTHONPATH', None)
        vision_env.pop('PYTHONHOME', None)
        children.start('perception', [sys.executable, '-u', str(ROOT / 'real_time_project/main.py'),
                       '--detector', 'yoloe', '--width', '640', '--height', '480', '--serial', args.serial,
                       '--prompt', args.prompt, '--conf', str(args.conf)], vision_env)
        print('[2/4] 인식 창에서 큐브를 확인하세요. 안정된 pose를 얻으면 자동으로 경로를 계산합니다.', flush=True)
        deadline = time.monotonic() + args.timeout
        while True:
            children.check()
            for msg in joints.packets():
                cur = msg
            for msg in poses.packets():
                T = stable.add(msg)
                if T is not None and fresh_joints(cur):
                    res = make_plan(kin, checker, T, cur['posj'], pargs, t_path)
                    path = save_plan(out, res)
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as pub:
                        pub.sendto(json.dumps(res, allow_nan=False).encode(), ('127.0.0.1', 5010))
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


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print('\n파이프라인 종료 요청.', flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'\nSTOP: {exc}', file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
