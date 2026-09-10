"""Offline recovery transitions; no camera, model weights or CUDA."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from FoundationPose.real_time_project.recovery_tracker_kimm import RecoveryTracker


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.detector, self.tracker = Mock(), Mock()
        self.mask = object()
        self.detector.infer.return_value = dict(success=True, mask=self.mask)
        self.frame = SimpleNamespace(rgb=object(), timestamp_ns=1_000_000_000)
        self.recovery = RecoveryTracker(self.detector, self.tracker, retry_interval=0)

    def test_register_same_frame_then_track_without_detector(self):
        result = self.recovery.process(self.frame)
        self.assertTrue(result['registered'])
        self.detector.infer.assert_called_once_with(self.frame.rgb, "rubik's cube")
        self.tracker.estimate.assert_called_once_with(self.frame, self.mask)
        # Long initial registration must not cause an endless re-registration loop.
        self.frame.timestamp_ns += 10_000_000_000
        self.recovery.process(self.frame)
        self.detector.infer.assert_called_once()
        self.tracker.estimate.assert_called_with(self.frame)

    def test_lost_retries_on_next_frame(self):
        self.recovery.process(self.frame)
        self.tracker.estimate.side_effect = ValueError('Tracking lost')
        result = self.recovery.process(self.frame)
        self.assertIsNone(result['pose'])
        self.assertFalse(self.recovery.tracking)
        self.tracker.estimate.side_effect = None
        fresh = SimpleNamespace(rgb=object(), timestamp_ns=2_000_000_000)
        self.assertTrue(self.recovery.process(fresh)['registered'])
        self.detector.infer.assert_called_with(fresh.rgb, "rubik's cube")
        self.tracker.estimate.assert_called_with(fresh, self.mask)

    def test_miss_and_failed_registration_retry(self):
        self.detector.infer.return_value = dict(success=False, mask=None)
        self.assertIsNone(self.recovery.process(self.frame)['pose'])
        self.tracker.estimate.assert_not_called()
        self.detector.infer.return_value = dict(success=True, mask=self.mask)
        self.tracker.estimate.side_effect = ValueError('invalid depth')
        self.assertEqual(self.recovery.process(self.frame)['error'], 'invalid depth')
        self.assertFalse(self.recovery.tracking)
        self.assertEqual(self.recovery.search_attempt, 2)
        self.tracker.estimate.side_effect = None
        self.assertTrue(self.recovery.process(self.frame)['registered'])
        self.assertEqual(self.recovery.search_attempt, 3)
        self.recovery.reset()
        self.assertEqual(self.recovery.search_attempt, 0)

    def test_retry_interval_and_manual_reset(self):
        self.recovery.retry_interval = 0.5
        self.detector.infer.return_value = dict(success=False, mask=None)
        with patch('FoundationPose.real_time_project.recovery_tracker_kimm.time.perf_counter', return_value=10):
            self.recovery.process(self.frame)
            self.recovery.process(self.frame)
            self.detector.infer.assert_called_once()
            self.recovery.reset()
            self.recovery.process(self.frame)
            self.assertEqual(self.detector.infer.call_count, 2)

    def test_frame_gap_triggers_recovery(self):
        self.recovery.process(self.frame)
        self.recovery.process(self.frame)
        self.frame.timestamp_ns += 2_000_000_000
        self.assertTrue(self.recovery.process(self.frame)['registered'])
        self.assertEqual(self.detector.infer.call_count, 2)

    def test_watchdog_flags_suspect_then_lost_after_patience(self):
        self.recovery.validation_interval = 0.5
        self.recovery.loss_patience = 2
        self.detector.infer.return_value['bbox'] = [0, 0, 10, 10]
        self.recovery.process(self.frame)
        self.recovery.next_validation = 0
        self.detector.detect_bbox.return_value = dict(bbox=[0, 0, 10, 10])
        self.tracker.validate_geometry.side_effect = [None, ValueError('bbox mismatch')]

        # First disagreement: still tracking, just flagged suspect (patience=2).
        result = self.recovery.process(self.frame)
        self.assertIsNotNone(result['pose'])
        self.assertTrue(result['suspect'])
        self.assertTrue(self.recovery.tracking)

        # Second disagreement in a row exhausts patience -> real loss.
        self.frame.timestamp_ns += 10_000_000  # distinct frame; gap=0 would look stale
        self.recovery.next_validation = 0
        self.tracker.validate_geometry.side_effect = [None, ValueError('bbox mismatch')]
        result = self.recovery.process(self.frame)
        self.assertIsNone(result['pose'])
        self.assertFalse(self.recovery.tracking)
        self.assertEqual(self.detector.detect_bbox.call_count, 2)

    def test_bad_registration_geometry_does_not_publish_pose(self):
        self.recovery.validation_interval = 0.5
        self.detector.infer.return_value['bbox'] = [0, 0, 10, 10]
        self.tracker.validate_geometry.side_effect = ValueError('bad depth')
        result = self.recovery.process(self.frame)
        self.assertIsNone(result['pose'])
        self.assertFalse(result['registered'])
        self.assertFalse(self.recovery.tracking)

    def test_geometry_guard_depth_and_detector_position(self):
        import numpy as np
        from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker
        tracker = PoseTracker.__new__(PoseTracker)
        tracker.to_origin = np.eye(4)
        tracker.bbox = np.array([[-.0285]*3, [.0285]*3])
        frame = SimpleNamespace(K=np.array([[100,0,50],[0,100,50],[0,0,1]]),
                                depth=np.full((100,100), .48))
        pose = np.eye(4)
        pose[2,3] = .5
        tracker.validate_geometry(frame, pose, [40,40,60,60])
        with self.assertRaisesRegex(ValueError, 'DINO'):
            tracker.validate_geometry(frame, pose, [5,5,15,15])
        frame.depth[:] = 1.5
        with self.assertRaisesRegex(ValueError, 'depth'):
            tracker.validate_geometry(frame, pose)

    def test_low_score_is_soft_and_keeps_baseline(self):
        import numpy as np
        from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker
        tracker = PoseTracker.__new__(PoseTracker)
        tracker.est = Mock()
        tracker.est.track_one.return_value = np.eye(4)
        tracker.track_iterations = 2
        tracker.drift_score_ratio = 0.6
        tracker.score_ema_alpha = 0.2
        tracker.score_ema = tracker.registration_score = 100.0
        tracker._score = Mock(side_effect=[90.0, 50.0])
        frame = SimpleNamespace(rgb=None, depth=None, K=None, identifier='42')
        with patch('builtins.print') as output:
            tracker.estimate(frame)
            self.assertEqual(tracker.score_ema, 98.0)
            self.assertTrue(tracker.last_score_ok)
            # A low score does not raise: it's a soft signal, and the baseline
            # must not blend in the rejected score (else it would chase drift down).
            pose = tracker.estimate(frame)
            self.assertFalse(tracker.last_score_ok)
            self.assertEqual(tracker.score_ema, 98.0)
        self.assertTrue(np.array_equal(pose, np.eye(4)))
        lines = '\n'.join(call.args[0] for call in output.call_args_list)
        self.assertIn('threshold=58.8000', lines)
        self.assertIn('decision=SOFT_LOW', lines)

    def test_runtime_errors_are_not_silently_retried(self):
        self.detector.infer.side_effect = RuntimeError('CUDA out of memory')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.recovery.process(self.frame)


if __name__ == '__main__':
    unittest.main()
