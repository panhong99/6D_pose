"""Single-object RGB-D pose estimation. GPU work stays on the executor thread."""
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation


class PoseNode(Node):
    def __init__(self):
        super().__init__('foundationpose')
        defaults = dict(
            foundationpose_dir='/home/panhong/pan/FoundationPose', mesh_file='',
            mesh_scale=1.0, target_label='cup', score_threshold=0.5,
            rgb_topic='/camera/camera/color/image_raw',
            depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
            info_topic='/camera/camera/color/camera_info', sync_slop=0.04)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        if not p('mesh_file'):
            raise ValueError('mesh_file parameter is required')
        sys.path.insert(0, str(Path(p('foundationpose_dir')).resolve()))
        import run_live_d455 as fp
        self.fp = fp
        mesh = fp.load_mesh(p('mesh_file'), p('mesh_scale'))
        self.est = fp.FoundationPose(
            mesh.vertices, mesh.vertex_normals, mesh=mesh,
            scorer=fp.ScorePredictor(), refiner=fp.PoseRefinePredictor(),
            glctx=fp.dr.RasterizeCudaContext(), debug=0,
            debug_dir=str(Path('/tmp/foundationpose_ros_debug')))
        self.detector, self.categories = fp.make_maskrcnn()
        self.bridge = CvBridge()
        self.info = None
        self.registered = False
        self.publisher = self.create_publisher(PoseStamped, '/foundationpose/pose', 10)
        self.create_service(Trigger, '/foundationpose/reset', self.reset)
        self.create_subscription(CameraInfo, p('info_topic'), self.on_info, qos_profile_sensor_data)
        self.rgb_sub = Subscriber(self, Image, p('rgb_topic'), qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, p('depth_topic'), qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=2, slop=p('sync_slop'))
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info('Ready for aligned RGB-D and color CameraInfo')

    def on_info(self, msg):
        self.info = msg

    def reset(self, request, response):
        self.registered = False
        self.est.pose_last = None
        response.success = True
        response.message = 'Next synchronized frame will re-detect and register'
        return response

    def on_frame(self, color_msg, depth_msg):
        if self.info is None:
            return
        try:
            if (self.info.width, self.info.height) != (color_msg.width, color_msg.height):
                raise ValueError('CameraInfo resolution differs from RGB')
            if self.info.header.frame_id != color_msg.header.frame_id:
                raise ValueError('CameraInfo must describe the RGB optical frame')
            rgb = np.ascontiguousarray(self.bridge.imgmsg_to_cv2(color_msg, 'rgb8'))
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
            if depth_msg.encoding == '16UC1':
                depth *= 0.001  # ROS depth convention: uint16 millimetres
            elif depth_msg.encoding != '32FC1':
                raise ValueError('Depth must be 16UC1 (mm) or 32FC1 (m)')
            if depth.shape != rgb.shape[:2]:
                raise ValueError('Depth must be aligned to RGB with matching dimensions')
            depth[~np.isfinite(depth) | (depth < 0.001)] = 0
            K = np.asarray(self.info.k, dtype=np.float32).reshape(3, 3)
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError('Invalid intrinsics')
            if not self.registered:
                mask = self.fp.maskrcnn_mask(
                    rgb, self.detector, self.categories,
                    self.get_parameter('target_label').value,
                    self.get_parameter('score_threshold').value)
                mask = (mask > 0) & (depth > 0)
                if mask.sum() < 100:
                    raise ValueError('Insufficient valid target depth for registration')
                pose = self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
                if self.est.pose_last is None:
                    raise ValueError('Registration did not initialize tracking')
                self.registered = True
            else:
                pose = self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)
            if not np.isfinite(pose).all():
                raise ValueError('Non-finite pose')
            msg = PoseStamped()
            msg.header = color_msg.header
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pose[:3, 3])
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            (msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
            self.publisher.publish(msg)
        except Exception as exc:
            self.registered = False
            self.est.pose_last = None
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
