"""Robot side (ROS2 terminal, system Python, NOT conda): stream the M1013 tool-flange pose over UDP.

The capture script (conda env) receives it on 127.0.0.1:5006.  Needs the Doosan bringup running:

  source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash
  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=<robot_ip> port:=12345 model:=m1013
  python3 calibration/ros2_flange_udp.py            # in a second ROS2 terminal

Read-only: it only calls the services get_current_tool_flange_posx (base frame, tcp = 0) and
get_current_posj, never moves the robot.  The services are called directly with rclpy (the
DSR_ROBOT2 wrapper blocked on its first call in this build).  Service names in this build:
  /<robot_id>/<controller_ns>/aux_control/get_current_tool_flange_posx   (controller_ns = dsr_controller2)
Output JSON: {"posx": [x, y, z (mm), a, b, c (deg, ZYZ)], "posj": [j1..j6 (deg)], "t": unix_time}
"""
import argparse
import json
import socket
import time

import rclpy
from dsr_msgs2.srv import GetCurrentPosj, GetCurrentToolFlangePosx

DR_BASE = 0


def call(node, client, request, timeout):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result() if future.done() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--robot_id', default='dsr01', help='namespace of the running bringup')
    p.add_argument('--model', default='m1013')
    p.add_argument('--controller_ns', default='dsr_controller2',
                   help="sub-namespace holding aux_control/* services in this Doosan build ('' = none)")
    p.add_argument('--port', type=int, default=5006)
    p.add_argument('--rate', type=float, default=20.0)
    a = p.parse_args()

    rclpy.init()
    node = rclpy.create_node('flange_pose_udp')
    prefix = '/' + '/'.join(x for x in (a.robot_id, a.controller_ns) if x)
    flange = node.create_client(GetCurrentToolFlangePosx, f'{prefix}/aux_control/get_current_tool_flange_posx')
    posj = node.create_client(GetCurrentPosj, f'{prefix}/aux_control/get_current_posj')
    for client in (flange, posj):
        if not client.wait_for_service(timeout_sec=10.0):
            raise SystemExit(f'service {client.srv_name} not available (is the Doosan bringup running?)')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f'streaming tool-flange posx of {prefix} to 127.0.0.1:{a.port} ...', flush=True)
    try:
        while rclpy.ok():
            req = GetCurrentToolFlangePosx.Request()
            req.ref = DR_BASE
            fx = call(node, flange, req, 1.0)
            jx = call(node, posj, GetCurrentPosj.Request(), 1.0)
            if fx is not None and jx is not None and fx.success and jx.success and len(fx.pos) == 6 and len(jx.pos) == 6:
                sock.sendto(json.dumps({'posx': [float(v) for v in fx.pos], 'posj': [float(v) for v in jx.pos],
                                        't': time.time()}).encode(), ('127.0.0.1', a.port))
            time.sleep(1.0 / a.rate)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
