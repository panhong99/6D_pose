"""ROS 2 -> Isaac Sim bridge for the Doosan M1013 USD articulation.

This is a simulation-only bridge. It does not connect to a real robot.

Input topic (compatible with the Doosan M1013 controller config):
  /dsr_moveit_controller/joint_trajectory (trajectory_msgs/JointTrajectory)

Run with Isaac Sim's Python:
  /home/panhong/isaacsim/python.sh isaacsim_ros2_m1013_bridge.py
"""

import argparse
import sys

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
from pxr import Sdf, UsdGeom, UsdLux, UsdShade

from isaacsim.core.experimental.prims import Articulation

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState


ROBOT_USD = "/home/panhong/pan/doosan-robot2/dsr_description2/usd/m1013.usd"
DOOSAN_JOINTS = [f"joint_{i}" for i in range(1, 7)]


class M1013RosBridge(Node):
    def __init__(self, articulation, command_topic):
        super().__init__("isaacsim_m1013_bridge")
        self.articulation = articulation
        self.dof_names = list(articulation.dof_names)
        self.dof_index = {name: i for i, name in enumerate(self.dof_names)}
        self.latest_target = None

        self.subscription = self.create_subscription(
            JointTrajectory,
            command_topic,
            self.trajectory_callback,
            10,
        )
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / 30.0, self.publish_state)

        self.get_logger().info(f"USD DOFs: {self.dof_names}")
        self.get_logger().info(f"Listening on: {command_topic}")

    def trajectory_callback(self, msg):
        if not msg.joint_names or not msg.points:
            self.get_logger().warning("Ignoring empty JointTrajectory")
            return

        point = msg.points[-1]
        if len(point.positions) != len(msg.joint_names):
            self.get_logger().error("positions and joint_names have different lengths")
            return

        target = np.asarray(self.articulation.get_dof_positions(), dtype=np.float32).reshape(-1)
        unknown = []
        for name, position in zip(msg.joint_names, point.positions):
            if name not in self.dof_index:
                unknown.append(name)
                continue
            target[self.dof_index[name]] = float(position)

        if unknown:
            self.get_logger().error(f"Unknown USD joint names: {unknown}")
            return

        self.latest_target = target
        self.articulation.set_dof_position_targets(target.reshape(1, -1))
        self.get_logger().info(
            "Applied trajectory target (rad): "
            + np.array2string(target, precision=3, suppress_small=True)
        )

    def publish_state(self):
        positions = np.asarray(self.articulation.get_dof_positions(), dtype=float).reshape(-1)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.dof_names
        msg.position = positions.tolist()
        self.state_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command-topic",
        default="/dsr_moveit_controller/joint_trajectory",
        help="JointTrajectory topic to subscribe to",
    )
    args, ros_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ros_args

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World", "Xform")

    # Use a thin dark box instead of the default white ground plane so the
    # white Doosan mesh remains visible in the viewport and camera images.
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set((8.0, 8.0, 0.04))
    ground.AddTranslateOp().Set((0.0, 0.0, -0.02))
    ground_mat = UsdShade.Material.Define(stage, "/World/GroundMaterial")
    ground_shader = UsdShade.Shader.Define(stage, "/World/GroundMaterial/Shader")
    ground_shader.CreateIdAttr("UsdPreviewSurface")
    ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.035, 0.035, 0.035))
    ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    ground_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    ground_mat.CreateSurfaceOutput().ConnectToSource(ground_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(250.0)

    robot_prim = stage.DefinePrim("/World/m1013", "Xform")
    robot_prim.GetReferences().AddReference(ROBOT_USD)
    robot = Articulation("/World/m1013")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        simulation_app.update()

    rclpy.init(args=ros_args)
    node = M1013RosBridge(robot, args.command_topic)

    try:
        while simulation_app.is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            simulation_app.update()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
