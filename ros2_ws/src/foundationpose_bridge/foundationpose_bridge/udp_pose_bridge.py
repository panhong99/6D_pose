"""Receive CAD-to-camera poses without importing FoundationPose or CUDA."""
import json
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation


def packet_to_pose(data, default_frame_id):
    packet = json.loads(data)
    if not isinstance(packet, dict):
        raise ValueError('Expected a JSON object')
    frame_id = packet.get('frame_id', default_frame_id)
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError('Invalid frame_id')
    if not isinstance(packet.get('frame_identifier'), str):
        raise ValueError('Expected a string frame_identifier')
    stamp = packet['timestamp']
    if not isinstance(stamp, dict):
        raise ValueError('Expected timestamp object')
    sec, nanosec = stamp['sec'], stamp['nanosec']
    if (type(sec) is not int or type(nanosec) is not int
            or not 0 <= sec < 2**31 or not 0 <= nanosec < 1_000_000_000):
        raise ValueError('Invalid timestamp')
    pose = np.asarray(packet['pose'])
    if (pose.shape != (4, 4) or pose.dtype.kind not in 'iuf'
            or not np.isfinite(pose).all()):
        raise ValueError('Expected finite numeric 4x4 pose')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError('Invalid homogeneous pose')
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3, rtol=0)):
        raise ValueError('Invalid rotation matrix')
    quaternion = Rotation.from_matrix(rotation).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pose[:3, 3])
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = map(float, quaternion)
    return msg


class UdpPoseBridge(Node):
    def __init__(self):
        super().__init__('foundationpose_udp_bridge')
        self.socket = None
        try:
            self.declare_parameter('udp_host', '127.0.0.1')
            self.declare_parameter('udp_port', 5005)
            self.declare_parameter('frame_id', 'camera_color_optical_frame')
            self.default_frame_id = self.get_parameter('frame_id').value
            self.publisher = self.create_publisher(PoseStamped, '/foundationpose/pose', 10)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setblocking(False)
            address = (self.get_parameter('udp_host').value,
                       self.get_parameter('udp_port').value)
            self.socket.bind(address)
            self.create_timer(0.01, self.receive)
            self.get_logger().info(f'UDP {self.socket.getsockname()} -> /foundationpose/pose')
        except Exception:
            self.destroy_node()
            raise

    def receive(self):
        # Bound each timer callback even if a sender continuously floods the socket.
        for _ in range(64):
            try:
                data, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
                break
            try:
                msg = packet_to_pose(data, self.default_frame_id)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                self.get_logger().warning(f'Ignoring invalid pose packet: {exc}',
                                          throttle_duration_sec=5.0)
                continue
            self.publisher.publish(msg)

    def destroy_node(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UdpPoseBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
