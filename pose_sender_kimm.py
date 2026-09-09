"""Send mesh-to-camera poses to the existing ROS bridge; no ROS imports."""
import json
import socket

import numpy as np


class PoseSender:
    def __init__(self, host='127.0.0.1', port=5005,
                 frame_id='camera_color_optical_frame'):
        if not frame_id or not 1 <= port <= 65535:
            raise ValueError('A frame_id and UDP port in 1..65535 are required')
        self.address = (host, port)
        self.frame_id = frame_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pose, frame_identifier, timestamp_ns):
        pose = np.asarray(pose)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Expected a finite 4x4 pose')
        sec, nanosec = divmod(int(timestamp_ns), 1_000_000_000)
        packet = dict(frame_id=self.frame_id,
                      frame_identifier=str(frame_identifier),
                      timestamp=dict(sec=sec, nanosec=nanosec),
                      pose=pose.tolist())
        self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), self.address)

    def close(self):
        self.socket.close()
