"""Run with foundationpose Python; checks contracts without CUDA or a camera."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker
from FoundationPose.real_time_project.sam_mask_kimm import SamMask
from FoundationPose.real_time_project.d455_source_kimm import D455Source
from FoundationPose.real_time_project.camera_rectify_kimm import rectification_maps
import pyrealsense2 as rs


class TrackingContractTest(unittest.TestCase):
    def test_inverse_distortion_rectification_rays(self):
        intr = rs.intrinsics()
        intr.width, intr.height = 32, 24
        intr.fx, intr.fy, intr.ppx, intr.ppy = 25., 26., 16., 12.
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = [0.08, -0.02, 0.001, -0.002, 0.003]
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        mx, my = rectification_maps(intr, K)
        for u, v in [(0, 0), (31, 23), (16, 12), (4, 19)]:
            ray = rs.rs2_deproject_pixel_to_point(intr, [float(mx[v, u]), float(my[v, u])], 1.0)
            projected = K @ ray
            np.testing.assert_allclose(projected[:2], [u, v], atol=2e-4)
        self.assertEqual(mx.dtype, np.float32)
        intr.coeffs = [0.] * 5
        self.assertEqual(rectification_maps(intr, K), (None, None))

    def setUp(self):
        self.tracker = PoseTracker.__new__(PoseTracker)
        self.tracker.est = Mock(pose_last=None)
        self.tracker.est.scorer.predict.return_value = (np.array([1.0]), None)
        self.tracker.register_iterations = 5
        self.tracker.track_iterations = 2
        self.tracker.drift_score_ratio = 0.6
        self.tracker.score_ema_alpha = 0.2
        self.tracker.score_ema = None
        self.tracker.last_score = None
        self.frame = SimpleNamespace(rgb=np.zeros((20, 20, 3), dtype=np.uint8),
                                     depth=np.ones((20, 20), dtype=np.float32), K=np.eye(3))

    def test_cannot_track_before_registration(self):
        with self.assertRaises(ValueError):
            self.tracker.estimate(self.frame)
        self.tracker.est.track_one.assert_not_called()

    def test_empty_depth_does_not_register(self):
        self.frame.depth[:] = 0
        with self.assertRaises(ValueError):
            self.tracker.estimate(self.frame, np.ones((20, 20), dtype=bool))
        self.tracker.est.register.assert_not_called()

    def test_registration_then_tracking_preserves_inputs_and_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, -0.2, 0.7]
        def register(**kwargs):
            self.tracker.est.pose_last = torch.tensor(pose, dtype=torch.float32)
            return pose
        self.tracker.est.register.side_effect = register
        self.tracker.est.track_one.return_value = pose
        mask = np.ones((20, 20), dtype=bool)
        np.testing.assert_array_equal(self.tracker.estimate(self.frame, mask), pose)
        kwargs = self.tracker.est.register.call_args.kwargs
        self.assertIs(kwargs['rgb'], self.frame.rgb)
        self.assertIs(kwargs['depth'], self.frame.depth)
        np.testing.assert_array_equal(kwargs['ob_mask'], mask)
        np.testing.assert_array_equal(self.tracker.estimate(self.frame), pose)
        self.assertEqual(self.tracker.est.track_one.call_args.kwargs['iteration'], 2)

    def test_nonfinite_pose_clears_tracking_state(self):
        self.tracker.est.pose_last = np.eye(4)
        self.tracker.est.track_one.return_value = np.full((4, 4), np.nan)
        with self.assertRaises(ValueError):
            self.tracker.estimate(self.frame)
        self.assertIsNone(self.tracker.est.pose_last)

    def test_register_early_return_cannot_start_tracking(self):
        self.tracker.est.register.return_value = np.eye(4)
        with self.assertRaises(ValueError):
            self.tracker.estimate(self.frame, np.ones((20, 20), dtype=bool))

    def test_invalid_box_rejected_before_sam(self):
        sam = SamMask.__new__(SamMask)
        sam.predictor = Mock()
        for box in ([5, 5, 2, 8], [0, 0, 21, 10], [0, np.nan, 10, 10]):
            with self.assertRaises(ValueError):
                sam.predict(self.frame.rgb, box)
        sam.predictor.set_image.assert_not_called()

    def test_camera_depth_units_invalid_values_and_owned_rgb(self):
        camera = D455Source.__new__(D455Source)
        camera.pipeline, camera.align = Mock(), Mock()
        camera.scale, camera.max_depth = 0.001, 3.0
        camera.map1 = None
        camera.K = np.eye(3, dtype=np.float32)
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        frames = camera.align.process.return_value
        frames.get_color_frame.return_value.get_data.return_value = rgb
        frames.get_color_frame.return_value.get_frame_number.return_value = 42
        frames.get_depth_frame.return_value.get_data.return_value = np.array(
            [[0, 1000], [2000, 4000]], dtype=np.uint16)
        with patch('d455_source_kimm.time.time_ns', return_value=1700000000123456789):
            frame = camera.read()
        np.testing.assert_allclose(frame.depth, [[0, 1], [2, 0]])
        rgb[:] = 255
        self.assertEqual(frame.rgb.max(), 0)
        self.assertEqual(frame.timestamp_ns, 1700000000123456789)
        self.assertEqual(frame.identifier, '42')


if __name__ == '__main__':
    unittest.main()
