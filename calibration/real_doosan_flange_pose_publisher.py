"""Publish the real Doosan M1013 flange pose through ROS 2.

Output:
  /dsr01/flange_pose (std_msgs/Float64MultiArray)
  data = [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]

The pose is returned by the Doosan API relative to DR_BASE.  This publisher
does not command motion.
"""

import json
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from dsr_msgs2.srv import GetCurrentToolFlangePosx

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m1013"
FLANGE_JSON_PATH = "/tmp/doosan_flange_pose.json"


class FlangePosePublisher(Node):
    def __init__(self):
        super().__init__("doosan_flange_pose_publisher")
        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/dsr01/flange_pose",
            10,
        )
        self.flange_client = self.create_client(
            GetCurrentToolFlangePosx,
            "aux_control/get_current_tool_flange_posx",
        )
        self.create_timer(0.1, self.publish_pose)
        self.get_logger().info("Publishing flange pose relative to DR_BASE")

    def publish_pose(self):
        if not self.flange_client.service_is_ready():
            self.get_logger().warning("Waiting for flange pose service")
            return

        request = GetCurrentToolFlangePosx.Request()
        request.ref = 0  # DR_BASE
        future = self.flange_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        if not future.done() or future.result() is None or not future.result().success:
            self.get_logger().warning("Failed to read flange pose service")
            return
        pose = future.result().pos

        values = [float(v) for v in pose]
        msg = Float64MultiArray()
        msg.data = values
        self.publisher.publish(msg)
        with open(FLANGE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"pose_mm_deg": values}, f)
        self.get_logger().info(
            "flange [x,y,z,a,b,c] = "
            + ", ".join(f"{v:.3f}" for v in values)
            + " [mm, deg]"
        )


def main():
    rclpy.init()
    node = FlangePosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
