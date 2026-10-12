"""Hardware-free regression checks for orchestration and motion boundaries."""
import contextlib
import io
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import cube_pipeline as pipeline
import real_pose_dryrun as dry
import ros2_pendant_mover as mover
from calib_utils import inv_T, load_json


def pose_packet(T, stamp):
    return dict(pose=T.tolist(), timestamp=dict(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)))


class Pipeline(unittest.TestCase):
    def test_stability_rejects_stale_repeated_invalid_and_gapped_frames(self):
        a = dry.parse_args([])
        stable = pipeline.StablePose(np.eye(4), a)
        now = time.time()
        T = np.eye(4)
        for i in range(8):
            result = stable.add(pose_packet(T, now - .2 + i * .01), now)
        self.assertIsNotNone(result)
        for packet in (pose_packet(T, now - 3), pose_packet(T, now - .13),
                       pose_packet(T * 2, now), dict(pose=[[float('nan')]])):
            self.assertIsNone(stable.add(packet, now))
            self.assertEqual(len(stable.history), 0)
        stable.add(pose_packet(T, now), now)
        stable.add(pose_packet(T, now + 1.1), now + 1.1)
        self.assertEqual(len(stable.history), 1)

    def test_no_joints_fallback_and_nonfinite_inputs_refused(self):
        now = time.time()
        self.assertFalse(pipeline.fresh_joints(None, now))
        self.assertFalse(pipeline.fresh_joints(dict(posj=[0] * 6, t=now - 2), now))
        self.assertFalse(pipeline.fresh_joints(dict(posj=[float('nan')] * 6, t=now), now))
        self.assertTrue(pipeline.fresh_joints(dict(posj=[0] * 6, t=now - .1), now))

    def test_default_has_no_execute_and_offline_cannot_enable_motion(self):
        args = pipeline.parse_args([])
        for leg in ('approach', 'return'):
            self.assertNotIn('--execute', pipeline.mover_command(args, Path('plan.json'), leg))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--offline-test', '--execute'])
        wp = [[0] * 6, [10] * 6]
        plan = dict(dry_run=True, accepted=True, offline_test=True, execution_allowed=False,
                    created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=.1,
                    legs=dict(approach=dict(waypoints_deg=wp, min_clearance_straight_m=.12)))
        args = mover.parse_args(['--plan', 'unused', '--leg', 'approach'])
        self.assertTrue(mover.validate(args, plan)[1])
        for option in ('--vel_deg_s', '--acc_deg_s2'):
            for value in ('3', 'nan'):
                a = mover.parse_args(['--plan', 'unused', '--leg', 'approach', option, value])
                clean_plan = dict(plan, offline_test=False, execution_allowed=True)
                self.assertTrue(mover.validate(a, clean_plan)[1])

    def run_with_fake_io(self, execute=False, refused=False):
        """Real planner + synthetic sensor streams; no process, ROS, socket or camera."""
        args = pipeline.parse_args([])
        args.execute = execute
        args.t_base_cam = dry.latest_t_base_cam()
        Tbc = np.asarray(load_json(dry.latest_t_base_cam())['T_base_cam'])
        Tbase = np.eye(4)
        Tbase[:3, 3] = [0.85, 0.50, .0285] if refused else [1.04, .05, .0285]
        Tcam = inv_T(Tbc) @ Tbase
        poses, joints = MagicMock(), MagicMock()
        now = time.time()
        poses.packets.side_effect = lambda: iter(pose_packet(Tcam, now - .2 + i * .01) for i in range(8))
        joints.packets.side_effect = lambda: iter([dict(posj=[0] * 6, t=time.time())])
        children = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(dry, 'DATA', Path(tmp)), \
                patch.object(pipeline, 'Children', return_value=children), \
                patch.object(pipeline, 'Receiver', side_effect=[poses, joints]), \
                patch.object(pipeline, 'ros_environment', return_value={}), \
                patch.object(pipeline.socket, 'socket') as udp, \
                patch.object(pipeline, 'wait_choice', return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            if refused:
                with self.assertRaisesRegex(RuntimeError, '경로 거부'):
                    pipeline.run(args)
            else:
                pipeline.run(args)
            children.close.assert_called_once()
            poses.sock.close.assert_called_once()
            joints.sock.close.assert_called_once()
            udp.return_value.__enter__.return_value.sendto.assert_called_once()
            return children

    def test_default_live_flow_plans_publishes_and_cleans_up_without_mover(self):
        children = self.run_with_fake_io()
        self.assertEqual(children.start.call_count, 3)
        children.run_mover.assert_not_called()

    def test_execute_routes_both_legs_through_existing_supervised_mover(self):
        children = self.run_with_fake_io(execute=True)
        self.assertEqual(children.run_mover.call_count, 2)
        for call in children.run_mover.call_args_list:
            command = call.args[0]
            self.assertIn('--execute', command)
            self.assertEqual(command[0], '/usr/bin/python3')
            self.assertIn('ros2_pendant_mover.py', command[2])

    def test_refused_plan_never_reaches_mover(self):
        children = self.run_with_fake_io(execute=True, refused=True)
        children.run_mover.assert_not_called()

    def test_process_failure_during_mover_interrupts_mover_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            children = pipeline.Children(Path(tmp))
            moving = MagicMock()
            moving.poll.return_value = None
            moving.wait.return_value = 130
            with patch.object(pipeline.subprocess, 'Popen', return_value=moving), \
                    patch.object(children, 'check', side_effect=[None, RuntimeError('perception stopped')]):
                with self.assertRaisesRegex(RuntimeError, 'perception stopped'):
                    children.run_mover(['fake-supervised-mover'], {})
            children.close()
            moving.send_signal.assert_called_once_with(pipeline.signal.SIGINT)

    def test_mover_rechecks_robot_after_go_before_sending_any_move(self):
        # Exercise the actual mover entrypoint with fake ROS clients only.
        node = MagicMock()
        current, motion, stop = MagicMock(), MagicMock(), MagicMock()
        node.create_client.side_effect = [current, motion, stop]
        current.call_async.side_effect = [
            SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, pos=[0] * 6)),
            SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, pos=[5] * 6))]
        fake_ros = SimpleNamespace(init=lambda: None, create_node=lambda _: node,
                                   spin_until_future_complete=lambda *a, **kw: None,
                                   ok=lambda: True, shutdown=lambda: None)
        srv = SimpleNamespace(GetCurrentPosj=SimpleNamespace(Request=SimpleNamespace),
                              MoveJoint=SimpleNamespace(Request=SimpleNamespace),
                              MoveStop=SimpleNamespace(Request=SimpleNamespace))
        plan = dict(dry_run=True, accepted=True, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    margin_m=.1, legs=dict(approach=dict(waypoints_deg=[[0] * 6, [10] * 6],
                                                       min_clearance_straight_m=.12)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text(json.dumps(plan))
            args = mover.parse_args(['--plan', str(path), '--leg', 'approach', '--execute'])
            with patch.dict('sys.modules', {'rclpy': fake_ros, 'dsr_msgs2': MagicMock(), 'dsr_msgs2.srv': srv}), \
                    patch.object(mover, 'parse_args', return_value=args), \
                    patch.object(mover, 'print_table'), \
                    patch('builtins.input', side_effect=[mover.OPENING, 'go']), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(SystemExit, 'robot moved while waiting'):
                mover.main()
            motion.call_async.assert_not_called()
            node.destroy_node.assert_called_once()


if __name__ == '__main__':
    unittest.main()
