"""Keep the fixed/wrist camera, YOLOE and FoundationPose warm between plate runs.

Run with the foundationpose interpreter.  This service never commands the robot.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from calib_core import cpath, load_config


ROOT = Path(__file__).resolve().parent.parent
STATE = Path('/dev/shm/6d_pose_perception_service.json')
SHARE = Path('/dev/shm/fixed_latest.npz')
LOG = Path('/tmp/6d_pose_perception_service.log')


def process_start(pid):
    """Linux process start tick, to avoid trusting a reused PID."""
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = stat.rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def live_state():
    try:
        state = json.loads(STATE.read_text())
        pid = int(state['pid'])
        if process_start(pid) != state['start_tick']:
            return None
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if str(ROOT / 'real_time_project/main.py') not in command:
            return None
        return state
    except (OSError, ValueError, KeyError, TypeError):
        return None


def specification(color):
    cfg = load_config()
    if color == 'white':
        prompt, mesh = cfg['white_plate']['prompt'], cpath(cfg['white_plate']['obj'])
    else:
        prompt, mesh = cfg['plate_rl']['prompt'], cpath(cfg['paths']['plate_obj'])
    return dict(prompt=prompt, mesh=str(Path(mesh).resolve()), conf=0.15,
                serial=cfg['cameras']['fixed_serial'], wrist_serial=cfg['cameras']['wrist_serial'])


def other_camera_owner():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode(errors='replace')
        except OSError:
            continue
        if str(ROOT / 'real_time_project/main.py') in command:
            return int(entry.name)
    return None


def start(color):
    existing = live_state()
    spec = specification(color)
    if existing:
        if all(existing.get(key) == value for key, value in spec.items()):
            print(f'Already warm: pid {existing["pid"]}, {color}')
            return
        raise RuntimeError(f'Existing perception pid {existing["pid"]} uses {existing.get("prompt")}; stop it before changing color')
    owner = other_camera_owner()
    if owner is not None:
        raise RuntimeError(f'Camera already owned by perception pid {owner}; wait for that run to finish')
    command = [sys.executable, '-u', str(ROOT / 'real_time_project/main.py'), '--detector', 'yoloe',
               '--width', '640', '--height', '480', '--serial', spec['serial'], '--prompt', spec['prompt'],
               '--conf', str(spec['conf']), '--mesh_file', spec['mesh'], '--wrist_serial', spec['wrist_serial'], '--no-display']
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME')}
    with LOG.open('ab') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    state = dict(pid=child.pid, start_tick=process_start(child.pid), color=color, **spec)
    if state['start_tick'] is None:
        raise RuntimeError('Perception exited before process identity could be recorded; check ' + str(LOG))
    begin = time.time()
    while time.time() - begin < 90:
        if child.poll() is not None:
            raise RuntimeError(f'Perception exited ({child.returncode}); check {LOG}')
        if SHARE.exists() and SHARE.stat().st_mtime > begin:
            tmp = STATE.with_suffix('.tmp')
            tmp.write_text(json.dumps(state))
            os.replace(tmp, STATE)
            print(f'Warm: pid {child.pid}, {color}, fixed camera frames ready; log {LOG}')
            return
        time.sleep(0.2)
    os.kill(child.pid, signal.SIGINT)
    raise RuntimeError(f'No new fixed camera frame in 90 s; check {LOG}')


def stop():
    state = live_state()
    if state is None:
        print('No live perception service')
        return
    os.kill(state['pid'], signal.SIGINT)
    for _ in range(50):
        if process_start(state['pid']) != state['start_tick']:
            STATE.unlink(missing_ok=True)
            print('Stopped')
            return
        time.sleep(0.1)
    print(f'SIGINT sent to pid {state["pid"]}; check status before using the cameras')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop'))
    parser.add_argument('--plate_color', choices=('red', 'white'), default='white')
    args = parser.parse_args()
    if args.action == 'start':
        start(args.plate_color)
    elif args.action == 'stop':
        stop()
    else:
        print(json.dumps(live_state(), indent=2))


if __name__ == '__main__':
    main()
