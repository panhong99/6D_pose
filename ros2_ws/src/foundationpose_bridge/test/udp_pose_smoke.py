"""GPU-free UDP -> ROS subscriber smoke test (run with ROS Python)."""
import json
import socket
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'FoundationPose'))
from pose_sender_kimm import PoseSender

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from foundationpose_bridge.udp_pose_bridge import UdpPoseBridge


def main():
    rclpy.init(args=['--ros-args', '-p', 'udp_port:=0'])
    bridge = UdpPoseBridge()
    subscriber = rclpy.create_node('udp_pose_smoke_subscriber')
    received = []
    subscription = subscriber.create_subscription(
        PoseStamped, '/foundationpose/pose', received.append, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(subscriber)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = bridge.socket.getsockname()
    pose_sender = PoseSender(*address)
    packet = {
        'frame_id': 'camera_color_optical_frame', 'frame_identifier': 'fake_001',
        'timestamp': {'sec': 1700000000, 'nanosec': 123456789},
        'pose': [[0, -1, 0, 0.1], [1, 0, 0, -0.2],
                 [0, 0, 1, 0.7], [0, 0, 0, 1]],
    }
    try:
        deadline = time.monotonic() + 5
        while bridge.publisher.get_subscription_count() == 0:
            executor.spin_once(timeout_sec=0.05)
            assert time.monotonic() < deadline, 'ROS discovery timed out'
        invalid = [b'{', json.dumps(dict(packet, pose=[[1]])).encode(),
                   json.dumps(dict(packet, pose=[[float('nan')]*4]*4)).encode(),
                   json.dumps(dict(packet, pose=[[float('inf')]*4]*4)).encode()]
        for data in invalid:
            sender.sendto(data, address)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert not received, 'Invalid packet was published'
        deadline = time.monotonic() + 5
        while not received:
            pose_sender.send(np.asarray(packet['pose']), packet['frame_identifier'],
                             1700000000123456789)
            executor.spin_once(timeout_sec=0.05)
            assert time.monotonic() < deadline, 'Pose reception timed out'
        msg = received[0]
        assert msg.header.frame_id == packet['frame_id']
        assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (1700000000, 123456789)
        assert (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) == (0.1, -0.2, 0.7)
        q = msg.pose.orientation
        assert abs(q.x) < 1e-8 and abs(q.y) < 1e-8
        assert abs(q.z - 2**-0.5) < 1e-8 and abs(q.w - 2**-0.5) < 1e-8
        print('PASS: UDP -> PoseStamped -> ROS subscriber; malformed/NaN/Inf rejected')
    finally:
        sender.close()
        pose_sender.close()
        executor.shutdown()
        subscriber.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
