"""ROS2 side of the DRY RUN (system Python, NOT conda): publish the planned target as NON-control topics.

  source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash
  python3 calibration/ros2_dryrun_publisher.py

Reads the JSON that calibration/real_pose_dryrun.py sends to 127.0.0.1:5010 and publishes
  /sixd/dryrun/target_pose        geometry_msgs/PoseStamped   pregrasp flange pose in the robot base frame
  /sixd/dryrun/target_joints_deg  std_msgs/Float64MultiArray  pregrasp joint angles j1..j6 [deg]
  /sixd/dryrun/waypoints_deg      std_msgs/Float64MultiArray  approach waypoints, row-major N x 6 [deg]
  /sixd/dryrun/status             std_msgs/String             JSON summary (accepted, reasons, clearances)
These topics are NOT read by the Doosan driver; it never publishes a command, calls a motion service, or touches
/dsr01.  A human reads the values and sets the pendant.
"""
import json
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray, String


class DryRunPublisher(Node):
    def __init__(self, port=5010):
        super().__init__('sixd_dryrun_publisher')
        self.pose = self.create_publisher(PoseStamped, '/sixd/dryrun/target_pose', 10)
        self.joints = self.create_publisher(Float64MultiArray, '/sixd/dryrun/target_joints_deg', 10)
        self.waypoints = self.create_publisher(Float64MultiArray, '/sixd/dryrun/waypoints_deg', 10)
        self.status = self.create_publisher(String, '/sixd/dryrun/status', 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.create_timer(0.05, self.poll)
        self.get_logger().info(f'DRY RUN publisher: UDP {port} -> /sixd/dryrun/* (no robot commands)')

    def poll(self):
        try:
            res = json.loads(self.sock.recv(1 << 20))
        except (BlockingIOError, ValueError):
            return
        legs = res.get('legs', {})
        self.status.publish(String(data=json.dumps(dict(
            accepted=res.get('accepted'), reasons=res.get('reasons'), cube_centre_m=res.get('cube_centre_m'),
            pregrasp_clearance_m=res.get('pregrasp_clearance_m'),
            approach_min_clearance_m=legs.get('approach', {}).get('min_clearance_straight_m'),
            return_min_clearance_m=legs.get('return', {}).get('min_clearance_straight_m'),
            dry_run=True))))
        if 'T_base_flange_pregrasp' not in res or not res.get('accepted'):
            return                                            # refused plans publish status only
        T = np.array(res['T_base_flange_pregrasp'])
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T[:3, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
        self.pose.publish(msg)
        self.joints.publish(Float64MultiArray(data=[float(v) for v in res['q_pregrasp_deg']]))
        wp = np.array(legs['approach']['waypoints_deg'], dtype=float)
        self.waypoints.publish(Float64MultiArray(data=wp.ravel().tolist()))


def main():
    rclpy.init()
    node = DryRunPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
