"""Tesollo DG-5F-S (20 DOF) first steps: connection check and one slow finger-by-finger bend.

    check : read-only. Does this PC have a 169.254.x address, does the hand answer on TCP 502, and (driver running)
            what are the joint angles now.  Sends nothing to the hand.
    wave  : (보스 runs it) fingers 1 (thumb) .. 5 one at a time: the two curl joints (_3, _4) bend by --bend rad from
            the start pose (clipped to +-CURL_LIMIT) over --move_s seconds and come back to the start pose.
            Enter per finger, q stops (the hand stays where it is).
    save NAME : read-only: the joint angles now -> tesollo/poses.yaml as NAME (pose shaped by hand in DGManager's
            teaching mode, then DGManager closed and the driver started again).
    go NAME   : (보스 runs it) all 20 joints to a saved pose; 'open' = all 0 (official MoveIt pose). One Enter, slow.

Needs the official driver running in another terminal (Developer Mode DIP switch, see README):
    source /opt/ros/jazzy/setup.bash && source ~/pan/tesollo_ws/install/setup.bash
    ros2 launch dg5f_s_driver dg5f_s_left_driver.launch.py delto_ip:=169.254.186.73
Run this file with the system python3 (ROS), not conda:
    source /opt/ros/jazzy/setup.bash && python3 tesollo/dg5fs.py check
"""
import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

JOINTS = [f'joint_{f}_{k}' for f in range(1, 6) for k in range(1, 5)]
CURL_LIMIT = 1.4                                  # rad; URDF allows +-1.57 on joints _3/_4, targets are clipped inside
MAX_BEND = 0.6                                    # rad; largest --bend, relative to the start pose
HOME_RATE = 0.25                                  # rad/s; go: the joint that moves most is no faster than this
POSES = Path(__file__).with_name('poses.yaml')    # saved hand poses (grasp, release, ...); 'open' = all 0 is built in


def tcp_ok(ip, port, timeout=2.0):
    """True if a TCP connection opens (closed at once, no bytes sent)."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError as e:
        print(f'  TCP {ip}:{port} -> {e}')
        return False


def link_local_addr():
    out = subprocess.run(['ip', '-br', 'addr'], capture_output=True, text=True).stdout
    return [w for line in out.splitlines() for w in line.split()[2:] if w.startswith('169.254.')]


class Hand:
    """Joint states in, joint trajectories out, for one driver namespace."""

    def __init__(self, side):
        import rclpy
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        rclpy.init()
        self.rclpy, self.ns = rclpy, f'dg5f_s_{side}'
        self.node = rclpy.create_node('dg5fs_tutorial')
        self.state = None
        self.node.create_subscription(JointState, f'/{self.ns}/joint_states', self._on_state, 10)
        self.pub = self.node.create_publisher(JointTrajectory, f'/{self.ns}/{self.ns}_controller/joint_trajectory', 10)

    def _on_state(self, msg):
        self.state = dict(zip(msg.name, msg.position))

    def read(self, timeout=3.0):
        self.state, t0 = None, time.time()
        while self.state is None and time.time() - t0 < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.state is None:
            return None
        missing = [j for j in JOINTS if j not in self.state]
        if missing:
            sys.exit(f'joint_states lacks {missing} (15-DOF hand? this tutorial is for the 20-DOF one)')
        return [self.state[j] for j in JOINTS]

    def go(self, q, move_s):
        """One trajectory point, reached in move_s seconds; waits until it should be there."""
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg = JointTrajectory(joint_names=JOINTS)
        msg.points = [JointTrajectoryPoint(positions=[float(v) for v in q],
                                           time_from_start=Duration(sec=int(move_s), nanosec=int(move_s % 1 * 1e9)))]
        self.pub.publish(msg)
        t0 = time.time()
        while time.time() - t0 < move_s + 0.5:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def show(q):
    for f in range(5):
        print(f'  finger {f + 1}' + ('(thumb)' if f == 0 else '       ') + ': ' + '  '.join(f'{v:+.2f}' for v in q[4 * f:4 * f + 4]))


def check(a):
    addrs = link_local_addr()
    print(f'1) PC 169.254.x address: {addrs or "none"}')
    if not addrs:
        print('   -> the PC cannot reach the hand; add one (until reboot):\n'
              '      sudo ip addr add 169.254.186.10/24 dev enp5s0')
    try:
        hand = Hand(a.side)
    except ImportError:
        hand = None
    q = hand.read() if hand else None
    route = subprocess.run(['ip', 'route', 'get', a.ip], capture_output=True, text=True).stdout.split('\n')[0]
    if ' via ' in route:                          # (10/26) via the site gateway something accepts any TCP: not the hand
        print(f'2) hand TCP {a.ip}:{a.port}: not tried - the route goes through a gateway ({route.strip()}), '
              'the hand must be on this cable directly')
    elif q is not None:                           # the hand takes one client: the running driver holds it
        print(f'2) hand TCP {a.ip}:{a.port}: held by the driver (not tried; a 2nd client is refused)')
    elif tcp_ok(a.ip, a.port):
        print(f'2) hand TCP {a.ip}:{a.port}: open')
    else:
        print(f'2) hand TCP {a.ip}:{a.port}: no answer\n'
              '   -> power, Ethernet cable, IP; green LED blinking = no socket yet (normal before the driver starts)')
    if hand is None:
        print('3) joint angles: rclpy missing - run with the ROS python (source /opt/ros/jazzy/setup.bash; python3 ...)')
        return
    hand.close()
    if q is None:
        print(f'3) /dg5f_s_{a.side}/joint_states: nothing in 3 s (driver not running, or other side?)')
    else:
        print(f'3) joint angles now [rad] (joints _1.._4 per finger):')
        show(q)


def wave(a):
    if not 0 < abs(a.bend) <= MAX_BEND:
        sys.exit(f'--bend must be in [-{MAX_BEND}, {MAX_BEND}] rad, not 0')
    if a.move_s < 1.5:
        sys.exit('--move_s must be >= 1.5 s')
    hand = Hand(a.side)
    start = hand.read()
    if start is None:
        sys.exit(f'no /dg5f_s_{a.side}/joint_states - start the driver first (see the top of this file)')
    print('start pose [rad]:')
    show(start)
    try:
        for f in range(5):
            name = 'finger 1 (thumb)' if f == 0 else f'finger {f + 1}'
            ans = input(f'\n{name}: bend joints _3/_4 by {a.bend:+.2f} rad in {a.move_s:.0f} s and back. Enter = go, q = stop: ')
            if ans.strip().lower() == 'q':
                break
            bent = list(start)
            for j in (4 * f + 2, 4 * f + 3):
                bent[j] = min(max(start[j] + a.bend, -CURL_LIMIT), CURL_LIMIT)
            hand.go(bent, a.move_s)
            q = hand.read()
            if q is not None:
                print(f'  reached: ' + '  '.join(f'{v:+.2f}' for v in q[4 * f:4 * f + 4]))
            hand.go(start, a.move_s)
        print('\ndone - the hand is back at the start pose')
    finally:
        hand.close()


def load_poses():
    """{name: 20 joint angles [rad]}: 'open' (all 0, the official MoveIt pose) + the saved ones in POSES."""
    import yaml
    saved = yaml.safe_load(POSES.read_text()) if POSES.exists() else {}
    return {'open': [0.0] * len(JOINTS), **{k: v['joints_rad'] for k, v in (saved or {}).items()}}


def save(a):
    """Read-only on the hand: the joint angles now -> POSES[name] (taught by hand in DGManager, driver back on)."""
    import yaml
    if a.name == 'open':
        sys.exit("'open' is fixed (all 0)")
    hand = Hand(a.side)
    q = hand.read()
    hand.close()
    if q is None:
        sys.exit(f'no /dg5f_s_{a.side}/joint_states - start the driver first (see the top of this file)')
    show(q)
    saved = (yaml.safe_load(POSES.read_text()) if POSES.exists() else None) or {}
    saved[a.name] = dict(joints_rad=[round(v, 4) for v in q], saved=time.strftime('%Y-%m-%d %H:%M:%S'), side=a.side)
    POSES.write_text(yaml.safe_dump(saved, sort_keys=False))
    print(f'saved as "{a.name}" in {POSES}')


def go(a):
    """(보스 runs it) all 20 joints to a saved pose (or 'open'), slowly: the worst joint <= HOME_RATE rad/s, one Enter."""
    poses = load_poses()
    if a.name not in poses:
        sys.exit(f'no pose "{a.name}"; saved: {list(poses)}')
    target = poses[a.name]
    hand = Hand(a.side)
    start = hand.read()
    if start is None:
        sys.exit(f'no /dg5f_s_{a.side}/joint_states - start the driver first (see the top of this file)')
    print('now [rad]:')
    show(start)
    print(f'"{a.name}" [rad]:')
    show(target)
    far = max(abs(t - v) for t, v in zip(target, start))
    move_s = max(a.move_s, far / HOME_RATE)
    try:
        ans = input(f'\nto "{a.name}", largest change {far:.2f} rad in {move_s:.1f} s. Enter = go, q = stop: ')
        if ans.strip().lower() == 'q':
            return
        hand.go(target, move_s)
        q = hand.read()
        if q is not None:
            print('reached [rad]:')
            show(q)
    finally:
        hand.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('check', 'wave', 'save', 'go'))
    p.add_argument('name', nargs='?', default='open', help='save / go: pose name (grasp, release, open, ...)')
    p.add_argument('--side', choices=('right', 'left'), default='left')
    p.add_argument('--ip', default='169.254.186.73')   # 보스 10/26: left hand, /24, no gateway
    p.add_argument('--port', type=int, default=502)
    p.add_argument('--bend', type=float, default=0.5, help='wave: curl change from the start pose [rad] (sign = direction)')
    p.add_argument('--move_s', type=float, default=3.0, help='wave / go: seconds per move (go: at least the HOME_RATE time)')
    a = p.parse_args()
    {'check': check, 'wave': wave, 'save': save, 'go': go}[a.mode](a)


if __name__ == '__main__':
    main()
