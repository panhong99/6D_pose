"""Tesollo DG-5F-S left: force-controlled grasp through the vendor DGSDK (the DGManager GRASP button, from code).

(보스 runs it; Developer Mode DIP 2, ROS 2 hand driver and DGManager CLOSED: the hand takes one client.)
    ~/miniconda3/bin/python3 tesollo/dg5fs_sdk.py [--mode 5F_2FINGER_1_AND_2] [--force 3] [--pre pregrasp] [--fixed 3 4 5]
Steps, one Enter each (q = stop, the hand keeps its pose; release + stop always run at the end):
    1. move to --pre (tesollo/poses.yaml, or 'open') in --motion_ms
    2. grasp: the SDK force grasp with --mode / --force [N]; fingers in --fixed stay where they are (position mode)
    3. release, then 4. open
Joints [deg], currents [mA] and the hottest motor are printed after each step (watch the temperature: stall heats up).
Needs the vendor Python SDK outside this repo (USB 2.Software/.../SampleProgram/dg_python-main.zip, unzipped to
--sdk) and cffi (conda base python has it; the ROS / foundationpose pythons do not).
"""
import argparse
import sys
import time
from pathlib import Path

POSES = Path(__file__).with_name('poses.yaml')
JOINTS = 20


def load_pose(name):
    import math
    import yaml
    if name == 'open':
        return [0.0] * JOINTS
    saved = yaml.safe_load(POSES.read_text())
    if name not in saved:
        sys.exit(f'no pose "{name}" in {POSES}: {["open", *saved]}')
    return [math.degrees(v) for v in saved[name]['joints_rad']]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sdk', default='/home/pan/pan/tesollo_ws/dg_python-main/src')
    p.add_argument('--ip', default='169.254.186.73')
    p.add_argument('--port', type=int, default=502)
    p.add_argument('--mode', default='5F_2FINGER_1_AND_2', help='DGGraspMode name without the leading _ (5F_3FINGER, ...)')
    p.add_argument('--force', type=float, default=3.0, help='grasp force [N] (start low; DGManager range 0-20 N)')
    p.add_argument('--pre', default='pregrasp', help='pose before the grasp (poses.yaml name or open)')
    p.add_argument('--fixed', type=int, nargs='*', default=[3, 4, 5], help='fingers kept in position mode during the grasp')
    p.add_argument('--motion_ms', type=int, default=3000, help='time per pose move [ms] (slow)')
    a = p.parse_args()
    # manual: Grasp Force 0-200 x 0.1 N; fingertip max 19.4 N, long stall near max torque heats the motors
    if not 0.5 <= a.force <= 20.0:
        sys.exit('--force must be 0.5 .. 20 N')
    if a.motion_ms < 1500:
        sys.exit('--motion_ms must be >= 1500')
    sys.path.insert(0, a.sdk)
    from dgsdk import (DGSDK, CommunicationMode, ControlMode, DGGraspMode, DGModel, DGResult, GripperSetting,
                       GripperSystemSetting, ReceivedDataType)
    mode = getattr(DGGraspMode, '_' + a.mode, None)
    if mode is None:
        sys.exit(f'unknown --mode {a.mode}: {[m.name.lstrip("_") for m in DGGraspMode if m.name.startswith("_5F")]}')
    pre = load_pose(a.pre)

    # (10/27 real) ConnectToGripper segfaults unless the connected/disconnected callbacks are registered first (as the
    # vendor examples do); data is polled with get_gripper_data, no struct-by-value data callback
    g = DGSDK()

    def check(step, r):
        print(f'  {step}: {r.name}', flush=True)
        if r != DGResult.NONE:
            raise SystemExit(f'{step} failed: {r.name}')

    def show():
        time.sleep(0.3)
        try:
            d = g.get_gripper_data()
        except RuntimeError as e:
            print(f'  (no data: {e})')
            return
        joint, current, temp = list(d.joint[:JOINTS]), list(d.current[:JOINTS]), list(d.temperature[:JOINTS])
        for f in range(5):
            s = slice(4 * f, 4 * f + 4)
            print(f'  finger {f + 1}: ' + '  '.join(f'{v:+6.1f}' for v in joint[s]) + '  deg | '
                  + ' '.join(f'{c:6.0f}' for c in current[s]) + ' mA')
        t = max(temp)
        print(f'  hottest motor {t:.1f} C (J{temp.index(t) + 1})', flush=True)

    def ask(what):
        if input(f'\n{what}  Enter = go, q = stop: ').strip().lower() == 'q':
            raise KeyboardInterrupt

    check('system', g.set_gripper_system(GripperSystemSetting.create(
        ip=a.ip, port=a.port, control_mode=ControlMode.DEVELOPER, communication_mode=CommunicationMode.ETHERNET)))
    g.on_connected(lambda: print('  -> connected', flush=True))
    g.on_disconnected(lambda: print('  -> disconnected', flush=True))
    check('connect', g.connect_to_gripper())
    started = False
    try:
        check('option', g.set_gripper_option(GripperSetting.create(
            # (10/27 real) joint/finger count 20/5 -> SystemStart NOT_FOUND_MODEL; DGManager sends 0/0 and starts fine
            model=DGModel.DG_5F_S_LEFT, joint_count=0, finger_count=0,
            received_data_type=[ReceivedDataType.JOINT, ReceivedDataType.CURRENT, ReceivedDataType.TEMPERATURE, 0, 0, 0, 0])))
        check('start', g.system_start())
        started = True
        check('motion time', g.set_motion_time_all_equal(a.motion_ms))
        show()
        ask(f'1. move to "{a.pre}" in {a.motion_ms} ms.')
        check('move', g.move_joint_all(pre))
        time.sleep(a.motion_ms / 1000 + 0.5)
        show()
        ask(f'2. grasp {a.mode} at {a.force} N (fingers {a.fixed} hold position).')
        for f in range(1, 6):
            check(f'position mode finger {f}', g.set_position_mode_finger([1 if f in a.fixed else 0] * 4, f))
        check('grasp data', g.set_grasp_data(mode, a.force, 0, 1))
        check('grasp', g.grasp(1))
        time.sleep(2.0)
        show()
        ask('3. release.')
    except KeyboardInterrupt:
        print('\nstopped by 보스', flush=True)
    finally:
        if started:
            check('release', g.grasp(0))
            time.sleep(1.0)
            try:
                if input('\n4. open the hand (all 0)?  Enter = go, n = leave it: ').strip().lower() != 'n':
                    g.move_joint_all([0.0] * JOINTS)
                    time.sleep(a.motion_ms / 1000 + 0.5)
                    show()
            except EOFError:
                pass
            g.system_stop()
        g.disconnect_to_gripper()


if __name__ == '__main__':
    main()
