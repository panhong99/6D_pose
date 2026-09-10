"""Aligned, rectified RGB-D acquisition in meters, without ROS dependencies."""
from dataclasses import dataclass
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from FoundationPose.real_time_project.camera_rectify_kimm import rectification_maps


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    timestamp_ns: int
    identifier: str


class D455Source:
    def __init__(self, width=640, height=480, fps=30, serial='', max_depth=3.0):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        try:
            device = profile.get_device()
            print('Camera:', device.get_info(rs.camera_info.name), flush=True)
            self.scale = device.first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self.max_depth = max_depth
            self.map1 = self.map2 = None
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                              dtype=np.float32)
            # FoundationPose uses a pinhole K. Apply the same pixel mapping to RGB
            # and aligned depth so distortion does not silently violate that contract.
            self.map1, self.map2 = rectification_maps(intr, self.K)
        except Exception:
            self.pipeline.stop()
            raise

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth_frame:
            raise RuntimeError('Missing synchronized RGB-D frame')
        # Device timestamps may be relative to camera startup, not Unix epoch.
        # This v1 deliberately publishes host receipt time, not a claimed capture time.
        timestamp_ns = time.time_ns()
        rgb = np.asanyarray(color.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.scale
        if self.map1 is not None:
            rgb = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.map1, self.map2, cv2.INTER_NEAREST)
        depth[~np.isfinite(depth) | (depth < 0.001) | (depth > self.max_depth)] = 0
        return RGBDFrame(np.ascontiguousarray(rgb), depth, self.K.copy(),
                         timestamp_ns, str(color.get_frame_number()))

    def close(self):
        self.pipeline.stop()
