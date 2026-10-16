"""Hardware-free regression tests of the real-robot side (no camera, robot, ROS or Isaac).

  cd calibration && conda run -n foundationpose python -m unittest tests -v

  Kinematic   FK at q = 0 matches the controller, IK round trip, top-down grasp frame
  Transforms  posx round trip, eye-to-hand recovers a known camera
  Cell        measured frame layout, a plan in front of the robot is accepted, one inside a frame post is refused
  Mover       slow / accepted / fresh / small-step plans pass, everything else is refused; default speed is the slowest
  Trajectory  time parameterisation respects 2 deg/s
  Pipeline    pose stability gate, joint freshness, no motion without --execute, offline plans cannot move the robot,
              live flow with fake I/O, refused plans never reach the mover, mover re-checks the robot after `go`
"""
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

import demo_data
import pipeline
import ros2_nodes as nodes
from calib_core import DATA, inv_T, latest_t_base_cam, load_json, make_T, posx_to_T, solve_eye_to_hand, T_to_posx
from robot_model import (FRAME_CELL, Kinematics, build_checker, cube_yaw_deg, frame_layout, grasp_frames, measured_cell,
                         smooth_path, time_parameterize)

CELL = DATA / 'real_cell_measured.json'
OLD_CAMERA = make_T(np.eye(3), [1.5, 0.38, 0.115])           # D455 body position used by the original tests


def pose_packet(T, stamp):
    return dict(pose=T.tolist(), timestamp=dict(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)))


def plan_for(centre):
    p = pipeline.planner_params()
    kin = Kinematics()
    checker, _ = build_checker(kin, OLD_CAMERA, p.margin_m, p.tool_length_m, p.tool_radius_m, CELL)
    return pipeline.plan_dryrun(kin, checker, make_T(np.eye(3), centre), np.zeros(6), p), checker, p


class Kinematic(unittest.TestCase):
    def setUp(self):
        self.kin = Kinematics()

    def test_zero_pose_matches_the_real_robot(self):
        flange = self.kin.fk(np.zeros(6))[:3, 3] * 1000              # controller at q = 0: (0.02, 34.5, 1452.5) mm
        self.assertLess(np.abs(flange - [0.02, 34.5, 1452.5]).max(), 0.5)

    def test_ik_round_trip(self):
        q = np.radians([10, 40, 60, 5, 50, -20])
        self.assertLess(np.abs(self.kin.fk(self.kin.ik(self.kin.fk(q), q + 0.05)) - self.kin.fk(q)).max(), 1e-4)

    def test_top_down_grasp_frame(self):
        for yaw in (0.0, 30.0, -20.0):
            cube = np.array([1.0, 0.05, 0.0285])
            measured, _ = cube_yaw_deg(make_T(R.from_euler('z', yaw, degrees=True).as_matrix(), cube))
            self.assertAlmostEqual(measured, yaw, places=3)
            _, pre, _, q, _ = grasp_frames(self.kin, cube, measured, 0.0, 0.10)
            self.assertAlmostEqual(pre[2, 3], 0.1285, places=4)
            self.assertAlmostEqual(self.kin.fk(q)[2, 3], pre[2, 3], places=3)
            self.assertLess(abs(np.degrees(q[3])), 1.0)              # j4 ~ 0: tool points down


class Transforms(unittest.TestCase):
    def test_posx_round_trip(self):
        p = [967.75, 90.98, 173.33, 4.66, 168.52, 1.57]
        self.assertLess(np.abs(np.array(T_to_posx(posx_to_T(p))) - p).max(), 1e-6)

    def test_handeye_recovers_a_known_camera(self):
        rng = np.random.default_rng(0)
        Tbc = make_T(R.from_euler('xyz', [-125, 5, 178], degrees=True).as_matrix(), [1.5, 0.4, 0.1])
        Tfm = make_T(R.from_euler('xyz', [10, -5, 30], degrees=True).as_matrix(), [0.02, -0.01, 0.03])
        Tbf, Tcm = [], []
        for _ in range(20):
            a = make_T((R.from_euler('xyz', [180, 0, 0], degrees=True) * R.from_rotvec(rng.normal(0, .5, 3))).as_matrix(),
                       [rng.uniform(.4, .8), rng.uniform(-.3, .3), rng.uniform(.3, .7)])
            Tbf.append(a)
            Tcm.append(inv_T(Tbc) @ a @ Tfm)
        self.assertLess(np.linalg.norm((inv_T(Tbc) @ solve_eye_to_hand(Tbf, Tcm)[0])[:3, 3]), 1e-4)


class Cell(unittest.TestCase):
    def test_measured_frame_layout(self):
        up = {n: c for n, c, _ in frame_layout(measured_cell(CELL)) if n.startswith('upright')}
        self.assertEqual(sorted(round(v[0], 2) for v in up.values()), [0.85, 0.85, 1.65, 1.65])
        self.assertEqual(sorted(round(v[1], 2) for v in up.values()), [-0.56, -0.56, 0.56, 0.56])
        self.assertIn('table_size', FRAME_CELL)

    def test_plan_for_a_cube_in_front_of_the_robot(self):
        res, checker, p = plan_for([1.04, 0.05, 0.0285])
        self.assertTrue(res['accepted'], res['reasons'])
        for leg in res['legs'].values():
            self.assertGreaterEqual(leg['min_clearance_straight_m'], checker.margin)
            self.assertLessEqual(np.abs(np.diff(np.array(leg['waypoints_deg']), axis=0)).max(), p.max_segment_deg + 1e-6)

    def test_object_inside_the_frame_post_is_refused(self):
        self.assertFalse(plan_for([0.85, 0.50, 0.0285])[0]['accepted'])


class Mover(unittest.TestCase):
    def plan(self, **kw):
        plan = dict(dry_run=True, accepted=True, margin_m=0.10, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    legs=dict(approach=dict(waypoints_deg=[[0] * 6, [10, 10, 10, 0, 10, 0], [20, 20, 20, 0, 20, 0]],
                                            min_clearance_straight_m=0.12)))
        plan.update(kw)
        return plan

    def check(self, plan, *argv):
        return nodes.validate(nodes.mover_args(['--plan', 'x', '--leg', 'approach', *argv]), plan)[1]

    def test_accepts_a_good_slow_plan(self):
        self.assertEqual(self.check(self.plan()), [])

    def test_refuses_fast_unaccepted_stale_or_big_moves(self):
        self.assertTrue(self.check(self.plan(), '--vel_deg_s', '-1'))
        self.assertTrue(self.check(self.plan(accepted=False)))
        self.assertTrue(self.check(self.plan(created='2020-01-01 00:00:00')))
        big = self.plan()
        big['legs']['approach']['waypoints_deg'][1] = [80, 0, 0, 0, 0, 0]
        self.assertTrue(self.check(big))
        limit = self.plan()
        limit['legs']['approach']['waypoints_deg'][2] = [0, 0, 0, 0, 0, 359]
        self.assertTrue(self.check(limit, '--max_step_deg', '400'))

    def test_default_speed_is_the_slowest_setting(self):
        a = nodes.mover_args(['--plan', 'x', '--leg', 'approach'])
        self.assertLessEqual(a.vel_deg_s, 2.0)
        self.assertLessEqual(a.acc_deg_s2, 2.0)


class Trajectory(unittest.TestCase):
    def test_smooth_path_timing_respects_limits(self):
        wp = np.radians([[0] * 6, [20, 10, 30, 0, 20, 0], [10, 40, 20, 0, 10, 5]])
        frames = time_parameterize(smooth_path(wp), np.radians(2), np.radians(2), 1 / 30)
        self.assertLessEqual(np.degrees(np.abs(np.diff(frames, axis=0)).max() * 30), 2.0 * 1.05)
        self.assertTrue(np.allclose(frames[0], wp[0]) and np.allclose(frames[-1], wp[-1]))


class Pipeline(unittest.TestCase):
    def test_stability_rejects_stale_repeated_invalid_and_gapped_frames(self):
        stable = pipeline.StablePose(np.eye(4), pipeline.planner_params())
        now, T = time.time(), np.eye(4)
        for i in range(8):
            result = stable.add(pose_packet(T, now - .2 + i * .01), now)
        self.assertIsNotNone(result)
        for packet in (pose_packet(T, now - 3), pose_packet(T, now - .13), pose_packet(T * 2, now), dict(pose=[[float('nan')]])):
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
        self.assertIsNone(args.insert_force_limit_n)
        self.assertNotIn('--tool-force-limit-n', pipeline.mover_command(args, Path('plan.json'), 'insert',
                                                                        smooth=True, tool_force_limit_n=None))
        guarded = pipeline.mover_command(args, Path('plan.json'), 'insert', smooth=True, tool_force_limit_n=25.0)
        self.assertEqual(guarded[guarded.index('--tool-force-limit-n') + 1], '25.0')
        guarded_argv = ['--task', 'plate', '--execute', '--place', 'basket', '--insert', 'quadrant', '--quadrant', 'Q1',
                        '--insert-force-limit-n', '25']
        with patch.object(pipeline.sys, 'stdin', SimpleNamespace(isatty=lambda: True)):
            guarded_args = pipeline.parse_args(guarded_argv)
        self.assertEqual(guarded_args.insert_force_limit_n, 25.0)
        self.assertIn('--tool-force-limit-n', pipeline.mover_command(
            guarded_args, Path('plan.json'), 'insert', smooth=True, tool_force_limit_n=guarded_args.insert_force_limit_n))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--task', 'cube', *guarded_argv[1:]])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--offline-test', '--execute'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--insert-force-limit-n', '25'])  # opt-in is only allowed for a real quadrant execute
        plan = dict(dry_run=True, accepted=True, offline_test=True, execution_allowed=False, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    margin_m=.1, legs=dict(approach=dict(waypoints_deg=[[0] * 6, [10] * 6], min_clearance_straight_m=.12)))
        self.assertTrue(nodes.validate(nodes.mover_args(['--plan', 'unused', '--leg', 'approach']), plan)[1])
        for option in ('--vel_deg_s', '--acc_deg_s2'):
            for value in ('-1', 'nan'):                    # no upper limit since 2026-10-13 (boss); only > 0, finite
                a = nodes.mover_args(['--plan', 'unused', '--leg', 'approach', option, value])
                self.assertTrue(nodes.validate(a, dict(plan, offline_test=False, execution_allowed=True))[1])
        for value in ('0', '-1', 'nan'):
            a = nodes.mover_args(['--plan', 'unused', '--leg', 'insert', '--smooth', '--tool-force-limit-n', value])
            self.assertTrue(any('tool-force-limit' in problem for problem in nodes.validate(
                a, dict(plan, offline_test=False, execution_allowed=True))[1]))

    def test_basket_insert_policy_uses_expert_observation_and_mean_mapping(self):
        import importlib.util
        import torch
        expert_path = Path(__file__).resolve().parents[1] / 'rl' / 'basket' / 'expert.py'
        old_path = list(pipeline.sys.path)
        spec = importlib.util.spec_from_file_location('basket_expert_policy_contract_test', expert_path)
        expert = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expert)

        class MeanPolicy(torch.nn.Module):
            def forward(self, obs):
                mean = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], dtype=obs.dtype, device=obs.device)
                return mean[None].expand(obs.shape[0], -1), torch.ones_like(mean)[None].expand(obs.shape[0], -1)

        old_pol = dict(expert._CL_POL)
        try:
            expert._CL_POL.update(policy=MeanPolicy(), det=True, dr=False, dr_full=False,
                                  noseen=True, canon=True, seed=0, cache={})
            H = make_T(R.from_euler('xyz', [12, -8, 25], degrees=True).as_matrix(), [0.04, -0.01, 0.11])
            Tbb = make_T(R.from_euler('z', 3.0, degrees=True).as_matrix(), [0.807, -0.192, 0.04])
            got = pipeline.basket_insert_policy_params(expert, 'Q3', 7, H, Tbb)

            Hc = expert.canonical_hand(H)
            yaw_b = np.arctan2(Tbb[1, 0], Tbb[0, 0])
            expected_obs = np.concatenate(([0, 0, 1, 0], [7 / 11.0],
                [Tbb[0, 3] - 0.81, Tbb[1, 3] + 0.19, yaw_b], Hc[:3, 3],
                R.from_matrix(Hc[:3, :3]).as_rotvec(), [0, 0])).astype(np.float32)
            np.testing.assert_allclose(got['obs'], expected_obs, atol=1e-7, rtol=0)
            raw = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
            scaled = np.tanh(raw) * np.array(expert.INSERT_SCALE)
            np.testing.assert_allclose(got['act'], raw, atol=1e-7, rtol=0)
            for key, value in zip(('dx', 'dy', 'lean', 'dz', 'yaw'), scaled):
                self.assertAlmostEqual(got[key], float(value), places=6)

            adjusted, yaw = pipeline.apply_basket_insert_policy(
                dict(x_shift_m=0.03, y_extra_m=0.005, tilt_deg=6.25, centre_h=0.165), -1.0, got, 12.5)
            self.assertAlmostEqual(adjusted['x_shift_m'], 0.03 + scaled[0], places=6)
            self.assertAlmostEqual(adjusted['y_extra_m'], 0.005 + scaled[1], places=6)
            self.assertAlmostEqual(adjusted['tilt_deg'], 6.25 + scaled[2] * 12.5, places=6)
            self.assertAlmostEqual(adjusted['centre_h'], 0.165 + scaled[3], places=6)
            self.assertAlmostEqual(yaw, -1.0 + scaled[4], places=6)
        finally:
            expert._CL_POL.clear()
            expert._CL_POL.update(old_pol)
            pipeline.sys.path[:] = old_path

    def test_insert_policy_cli_is_opt_in_and_requires_real_sim_q_quadrant_mode(self):
        self.assertIsNone(pipeline.parse_args([]).insert_policy)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'policy.pt'
            ckpt.write_bytes(b'test placeholder')
            base = ['--task', 'plate', '--place', 'basket', '--insert', 'quadrant', '--quadrant', 'Q1', '--sim_q']
            valid = [*base, '--execute', '--insert_policy', str(ckpt)]
            with patch.object(pipeline.sys, 'stdin', SimpleNamespace(isatty=lambda: True)):
                args = pipeline.parse_args(valid)
            self.assertEqual(args.insert_policy, ckpt)
            invalid = (
                ['--task', 'plate', '--place', 'basket', '--insert', 'quadrant', '--quadrant', 'Q1', '--sim_q',
                 '--insert_policy', str(ckpt)],
                ['--task', 'plate', '--place', 'basket', '--insert', 'quadrant', '--quadrant', 'Q1', '--execute',
                 '--insert_policy', str(ckpt)],
                ['--task', 'plate', '--place', 'basket', '--insert', 'quadrant', '--execute', '--sim_q',
                 '--insert_policy', str(ckpt)],
            )
            for argv in invalid:
                with patch.object(pipeline.sys, 'stdin', SimpleNamespace(isatty=lambda: True)), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    pipeline.parse_args(argv)

    def test_force_guard_fake_wrench_series_requests_one_stop_on_norm_exceedance(self):
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

            def spin(self, timeout):
                self.now += timeout

        class PendingSpline:
            def done(self):
                return False

            def result(self):
                raise AssertionError('tripped spline must not be treated as completed')

        clock, stops = Clock(), []
        samples = iter(([10., 0, 0, 0, 0, 0], [24.9, 0, 0, 0, 0, 0],
                        [15., 20., 0.1, 0, 0, 0], [0., 0, 0, 0, 0, 0]))
        result = nodes.monitor_spline_force(PendingSpline(), clock.spin, lambda: next(samples),
                                            lambda: stops.append(clock()) or True, 25.0,
                                            timeout_s=10.0, poll_hz=20.0, clock=clock)
        self.assertEqual(result['status'], 'force_limit')
        self.assertGreater(result['force_n'], 25.0)  # norm([15,20,0.1]); torque is excluded
        self.assertEqual(len(stops), 1)

    def test_force_guard_equal_threshold_does_not_stop_and_bad_sensor_does(self):
        stops = []
        sample = nodes.force_limit_sample([15., 20., 0., 900., 0., 0.], 25.0,
                                          lambda: stops.append('trip') or True)
        self.assertFalse(sample['tripped'])  # strict >; only force xyz contributes
        self.assertEqual(stops, [])

        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

            def spin(self, timeout):
                self.now += timeout

        class PendingSpline:
            def done(self):
                return False

        clock = Clock()
        result = nodes.monitor_spline_force(PendingSpline(), clock.spin, lambda: None,
                                            lambda: stops.append('sensor') or True, 25.0,
                                            timeout_s=1.0, poll_hz=20.0, clock=clock)
        self.assertEqual(result['status'], 'sensor_error')
        self.assertEqual(stops, ['sensor'])

    def test_force_guard_monitor_exception_requests_stop(self):
        class PendingSpline:
            def done(self):
                return False

        stops = []
        result = nodes.monitor_spline_force(PendingSpline(), lambda _timeout: (_ for _ in ()).throw(RuntimeError('executor')),
                                            lambda: [0.] * 6, lambda: stops.append(True) or True,
                                            25.0, timeout_s=1.0)
        self.assertEqual(result['status'], 'monitor_error')
        self.assertEqual(stops, [True])

    def test_mover_force_guard_uses_fake_services_and_qstops_before_auto_continuation(self):
        node = MagicMock()
        current, motion, spline, stop, force = [MagicMock() for _ in range(5)]
        node.create_client.side_effect = [current, motion, spline, stop, force]
        completed = lambda value: SimpleNamespace(done=lambda: True, result=lambda: value)
        current.call_async.side_effect = lambda _req: completed(SimpleNamespace(success=True, pos=[0.] * 6))
        force.call_async.side_effect = [completed(SimpleNamespace(success=True, tool_force=[10., 0, 0, 0, 0, 0])),
                                        completed(SimpleNamespace(success=True, tool_force=[26., 0, 0, 0, 0, 0]))]
        spline.call_async.return_value = SimpleNamespace(done=lambda: False, result=lambda: None)
        stop.call_async.return_value = completed(SimpleNamespace(success=True))
        fake_ros = SimpleNamespace(init=lambda: None, create_node=lambda _name: node,
                                   spin_until_future_complete=lambda *_a, **_kw: None,
                                   spin_once=lambda *_a, **_kw: None, ok=lambda: True, shutdown=lambda: None)
        srv = SimpleNamespace(GetCurrentPosj=SimpleNamespace(Request=SimpleNamespace),
                              MoveJoint=SimpleNamespace(Request=SimpleNamespace),
                              MoveSplineJoint=SimpleNamespace(Request=SimpleNamespace),
                              MoveStop=SimpleNamespace(Request=SimpleNamespace),
                              GetToolForce=SimpleNamespace(Request=SimpleNamespace))
        plan = dict(dry_run=True, accepted=True, created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=.1,
                    legs=dict(insert=dict(waypoints_deg=[[0.] * 6, [.5] * 6, [1.] * 6], min_clearance_straight_m=.12)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text(json.dumps(plan))
            argv = ['--plan', str(path), '--leg', 'insert', '--execute', '--smooth', '--auto', '--opening_done',
                    '--tool-force-limit-n', '25']
            with patch.dict('sys.modules', {'rclpy': fake_ros, 'dsr_msgs2': MagicMock(), 'dsr_msgs2.srv': srv,
                                            'std_msgs': MagicMock(), 'std_msgs.msg': MagicMock()}), \
                    patch.object(nodes, 'print_table'), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(SystemExit, 'force-guarded spline aborted'):
                nodes.run_mover(argv)
        self.assertEqual(force.call_async.call_count, 2)  # baseline then one in-motion wrench
        spline.call_async.assert_called_once()
        stop.call_async.assert_called_once()
        self.assertEqual(stop.call_async.call_args.args[0].stop_mode, 1)  # Doosan DR_QSTOP
        motion.call_async.assert_not_called()
        node.destroy_node.assert_called_once()

    def run_with_fake_io(self, execute=False, refused=False):
        """Real planner + synthetic sensor streams; no process, ROS, socket or camera."""
        args = pipeline.parse_args([])
        args.execute = execute
        args.t_base_cam = latest_t_base_cam()
        Tbc = np.asarray(load_json(args.t_base_cam)['T_base_cam'])
        Tcam = inv_T(Tbc) @ make_T(np.eye(3), [0.85, 0.50, .0285] if refused else [1.04, .05, .0285])
        poses, joints, children = MagicMock(), MagicMock(), MagicMock()
        now = time.time()
        poses.packets.side_effect = lambda: iter(pose_packet(Tcam, now - .2 + i * .01) for i in range(8))
        joints.packets.side_effect = lambda: iter([dict(posj=[0] * 6, t=time.time())])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pipeline, 'DATA', Path(tmp)), \
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
            self.assertTrue(command[2].endswith('ros2_nodes.py') and command[3] == 'mover')

    def test_refused_plan_never_reaches_mover(self):
        self.run_with_fake_io(execute=True, refused=True).run_mover.assert_not_called()

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

    def _smooth_mover(self, start_pos):
        node = MagicMock()
        current, motion, spline, stop = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        node.create_client.side_effect = [current, motion, spline, stop]
        ok = lambda pos: SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, pos=pos))  # noqa: E731
        current.call_async.side_effect = [ok(start_pos), ok(start_pos), ok([10] * 6)]
        spline.call_async.return_value = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True))
        fake_ros = SimpleNamespace(init=lambda: None, create_node=lambda _: node, spin_until_future_complete=lambda *a, **kw: None,
                                   ok=lambda: True, shutdown=lambda: None)
        srv = SimpleNamespace(GetCurrentPosj=SimpleNamespace(Request=SimpleNamespace), MoveJoint=SimpleNamespace(Request=SimpleNamespace),
                              MoveSplineJoint=SimpleNamespace(Request=SimpleNamespace), MoveStop=SimpleNamespace(Request=SimpleNamespace))
        plan = dict(dry_run=True, accepted=True, created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=.1,
                    legs=dict(lift=dict(waypoints_deg=[[0] * 6, [5] * 6, [10] * 6], min_clearance_straight_m=.12)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text(json.dumps(plan))
            argv = ['--plan', str(path), '--leg', 'lift', '--execute', '--smooth', '--auto', '--opening_done', '--vel_deg_s', '6']
            with patch.dict('sys.modules', {'rclpy': fake_ros, 'dsr_msgs2': MagicMock(), 'dsr_msgs2.srv': srv,
                                            'std_msgs': MagicMock(), 'std_msgs.msg': MagicMock()}), \
                    patch.object(nodes, 'print_table'), patch('builtins.input', side_effect=AssertionError('no prompt')), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    nodes.run_mover(argv)
                    code = None
                except SystemExit as exc:
                    code = str(exc)
        return code, spline

    def test_smooth_leg_one_spline_only_from_the_planned_start(self):
        code, spline = self._smooth_mover([0] * 6)
        self.assertIsNone(code)
        self.assertEqual(spline.call_async.call_count, 1)                 # the whole leg is ONE movesj
        code, spline = self._smooth_mover([4] * 6)                       # 4 deg from the leg start
        self.assertIn('REFUSED', code)
        spline.call_async.assert_not_called()

    def test_mover_rechecks_robot_after_go_before_sending_any_move(self):
        node = MagicMock()                                        # actual mover entrypoint with fake ROS clients
        current, motion, spline, stop = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        node.create_client.side_effect = [current, motion, spline, stop]
        current.call_async.side_effect = [
            SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, pos=[0] * 6)),
            SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, pos=[5] * 6))]
        fake_ros = SimpleNamespace(init=lambda: None, create_node=lambda _: node, spin_until_future_complete=lambda *a, **kw: None,
                                   ok=lambda: True, shutdown=lambda: None)
        srv = SimpleNamespace(GetCurrentPosj=SimpleNamespace(Request=SimpleNamespace), MoveJoint=SimpleNamespace(Request=SimpleNamespace),
                              MoveSplineJoint=SimpleNamespace(Request=SimpleNamespace), MoveStop=SimpleNamespace(Request=SimpleNamespace))
        plan = dict(dry_run=True, accepted=True, created=time.strftime('%Y-%m-%d %H:%M:%S'), margin_m=.1,
                    legs=dict(approach=dict(waypoints_deg=[[0] * 6, [10] * 6], min_clearance_straight_m=.12)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text(json.dumps(plan))
            argv = ['--plan', str(path), '--leg', 'approach', '--execute']
            with patch.dict('sys.modules', {'rclpy': fake_ros, 'dsr_msgs2': MagicMock(), 'dsr_msgs2.srv': srv,
                                            'std_msgs': MagicMock(), 'std_msgs.msg': MagicMock()}), \
                    patch.object(nodes, 'print_table'), \
                    patch('builtins.input', side_effect=[nodes.OPENING, 'go']), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(SystemExit, 'robot moved while waiting'):
                nodes.run_mover(argv)
            motion.call_async.assert_not_called()
            node.destroy_node.assert_called_once()


class CameraPinTests(unittest.TestCase):
    def setUp(self):
        import camera_tools as camera
        self.camera = camera
        self.K = np.array([[500., 0, 160], [0, 500., 120], [0, 0, 1]])
        self.p0, self.p1 = np.array([.06, 0, 1.]), np.array([.06, .10, 1.])
        self.axis = np.array([1., 0, 0])

    def fixture(self, dx=0):
        import cv2
        image = np.full((240, 320, 3), 230, np.uint8)
        uv, _ = self.camera._pin_project(np.stack([self.p0, self.p1])+self.axis[None]*dx/1000, np.eye(4), self.K)
        cv2.line(image, tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int)), (30, 30, 30), 3)
        return image

    def match(self, image):
        return self.camera._pin_match(image, self.camera._pin_image_context(image, self.camera.load_config()),
                                      self.p0, self.p1, np.eye(4), self.K, self.axis)

    def test_known_ridge_shift_and_sign(self):
        for dx in (-4, 0, 4):
            r = self.match(self.fixture(dx))
            self.assertEqual(r['status'], 'detected')
            self.assertAlmostEqual(r['correction_mm'], dx, delta=0.5)
            self.assertGreaterEqual(r['continuity_ratio'], 0.45)
            self.assertLessEqual(r['continuity_ratio'], 1.0)

    def test_blank_floor_holes_not_pin(self):
        import cv2
        image = np.full((240, 320, 3), 230, np.uint8)
        for y in range(128, 176, 10):
            cv2.circle(image, (190, y), 2, (20, 20, 20), -1)
        result = self.match(image)
        self.assertNotEqual(result['status'], 'detected')
        self.assertIn('continuity_ratio', result)
        self.assertGreaterEqual(result['continuity_ratio'], 0.0)
        self.assertLessEqual(result['continuity_ratio'], 1.0)

    def test_pin_continuity_bridges_only_short_internal_crossings(self):
        import camera_tools as camera
        interrupted = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1], bool)
        self.assertEqual(camera._pin_longest_run(interrupted), 3)
        self.assertEqual(camera._pin_longest_run_bridged(interrupted, max_gap=1), 7)
        self.assertEqual(camera._pin_longest_run_bridged(interrupted, max_gap=2), 11)
        self.assertEqual(camera._pin_longest_run_bridged(interrupted, max_gap=0), 3)

    def test_plate_occlusion_is_not_a_measurement(self):
        image = self.fixture()
        image[:, 180:210] = (30, 30, 190)
        r = self.match(image)
        self.assertEqual(r['status'], 'occluded')
        self.assertIsNone(r['correction_mm'])

    def test_projected_2d_tine_offset_solver_and_singularity_guard(self):
        import camera_tools as camera
        J = np.array([[3.2, -0.7], [0.4, 2.6]])
        expected = np.array([6.0, -4.0])
        solved = camera._solve_pin_pixel_offset(J @ expected, J, 35.0)
        np.testing.assert_allclose(solved[0], expected, atol=1e-10)
        self.assertIsNone(camera._solve_pin_pixel_offset(np.array([100., 100.]), J, 35.0))
        self.assertIsNone(camera._solve_pin_pixel_offset(np.array([1., 1.]), np.array([[1., 1.], [1., 1.00001]]), 35.0))
        self.assertIsNotNone(camera._solve_pin_pixel_offset(J @ [6., -15.], J, 8.0, search_y_mm=20.0))
        self.assertIsNone(camera._solve_pin_pixel_offset(J @ [9., -15.], J, 8.0, search_y_mm=20.0))
        self.assertIsNone(camera._solve_pin_pixel_offset(J @ [6., -21.], J, 8.0, search_y_mm=20.0))

    def test_pin_pair_candidate_keeps_strict_disagreement_gate(self):
        import camera_tools as camera
        pair = camera._pin_pair_offset([3.0, 10.0], [4.0, 12.0])
        self.assertTrue(pair['consistent'])
        np.testing.assert_allclose(pair['candidate_xy_mm'], [3.5, 11.0])
        np.testing.assert_allclose(pair['disagreement_xy_mm'], [1.0, 2.0])
        boundary = camera._pin_pair_offset([3.0, 10.0], [4.0, 13.0])
        self.assertTrue(boundary['consistent'])  # Existing gate is inclusive at exactly 3 mm.
        rejected = camera._pin_pair_offset([3.0, 10.0], [4.0, 13.01])
        self.assertFalse(rejected['consistent'])
        np.testing.assert_allclose(rejected['candidate_xy_mm'], [3.5, 11.505])
        self.assertIsNone(camera._pin_pair_offset([np.nan, 0.0], [0.0, 0.0]))

    def test_2d_hough_match_recovers_synthetic_xy_shift(self):
        import cv2
        import camera_tools as camera
        K = np.array([[1500., 0, 320], [0, 1500., 240], [0, 0, 1.]])
        Tcb = np.eye(4)
        p0, p1 = np.array([.06, -.04, 1.]), np.array([.06, .06, 1.])
        axis_x, axis_y = np.array([1., 0, 0]), np.array([0., 1, 0])
        expected = np.array([6., -15.])
        image = np.full((480, 640, 3), 230, np.uint8)
        moved = np.stack([p0, p1]) + axis_x[None]*expected[0]/1000 + axis_y[None]*expected[1]/1000
        uv, _ = camera._pin_project(moved, Tcb, K)
        cv2.line(image, tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int)), (30, 30, 30), 3)
        result = camera._pin_match(image, camera._pin_image_context(image, camera.load_config()),
                                   p0, p1, Tcb, K, axis_x, search_mm=8, axis_y=axis_y,
                                   search_radius_px=70, search_y_mm=20)
        self.assertEqual(result['status'], 'detected')
        np.testing.assert_allclose(result['correction_xy_mm'], expected, atol=2.)

    def test_2d_hough_no_candidate_returns_rejection_record(self):
        import camera_tools as camera
        image = np.full((480, 640, 3), 230, np.uint8)
        p0, p1 = np.array([.06, -.04, 1.]), np.array([.06, .06, 1.])
        result = camera._pin_match(image, camera._pin_image_context(image, camera.load_config()),
                                   p0, p1, np.eye(4),
                                   np.array([[1500., 0, 320], [0, 1500., 240], [0, 0, 1.]]),
                                   np.array([1., 0, 0]), axis_y=np.array([0., 1., 0.]),
                                   search_radius_px=70, search_y_mm=20.)
        self.assertEqual(result['status'], 'weak_line')
        self.assertIsNone(result['correction_mm'])
        self.assertEqual(len(result['model_uv']), 2)

    def test_gap_xy_offset_requires_both_edges_for_x(self):
        import camera_tools as camera
        detections = []
        for row, y, y_offset in (('a', -.0475, 4.), ('b', .0475, 4.)):
            for edge, x, x_offset in (('low_x', .10, 2.), ('high_x', .12, 2.)):
                detections.append(dict(status='detected', correction_xy_mm=[x_offset, y_offset], row=row,
                                       edge=edge, model_local_xy_m=[x, y]))
        full = camera._gap_xy_offset([.11, 0.], detections)
        self.assertEqual(full['status'], 'detected')
        self.assertAlmostEqual(full['x_mm'], 2.)
        self.assertAlmostEqual(full['y_mm'], 4.)
        partial = camera._gap_xy_offset([.11, 0.], detections[:-1])
        self.assertEqual(partial['status'], 'partial')
        self.assertIsNone(partial['x_mm'])
        self.assertAlmostEqual(partial['y_mm'], 4.)

    def test_checked_layout_gap_prior_uses_both_checked_rows(self):
        import camera_tools as camera
        entries = []
        for row, y, dy in (('a', 10., -8.), ('b', 30., 6.)):
            for number, x, dx in ((4, 100., 2.), (5, 118., 4.)):
                entries.append(dict(quadrant='Q2', image_row=row, number=number, visibility='complete',
                                    root_basket_xyz_mm=[x+dx, y+dy, -95.], root_model_x_mm=x,
                                    model_row_y_mm=y))
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp)/'audit.json'
            audit.write_text(json.dumps(dict(pins=entries)))
            result = camera._checked_gap_root_offset(audit, 'Q2', 4)
        self.assertEqual(result['status'], 'measured_prior')
        np.testing.assert_allclose(result['measured_minus_model_xy_mm'], [3., -1.])
        self.assertEqual(result['visible_checked_tines'], 4)

    def test_label_ray_intersection_recovers_known_pin_points(self):
        Tbb = np.eye(4)
        Tbb[:3, 3] = [.8, .15, .14]
        Tbc = np.eye(4)
        Tbc[:3, :3] = np.diag([1., -1., -1.])
        Tbc[:3, 3] = [.8, .15, .60]
        K = self.K
        for local in ([.08, -.10, -.144], [.055, -.10, -.029]):
            world = (Tbb @ np.r_[local, 1])[:3]
            uv = self.camera._pin_project(world[None], np.linalg.inv(Tbc), K)[0][0]
            found = self.camera._pixel_to_basket_plane(uv, local[2], Tbb, Tbc, K)
            np.testing.assert_allclose(found, local, atol=1e-9)

    def test_visible_comb_root_uses_rail_height_not_floor(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rl'))
        import basket_env as be
        floor = be.B['full']['floor_z']
        root_z = floor + be.B['pins']['rail_z']
        tip_z = floor + be.B['pins']['height']*np.cos(np.radians(be.B['pins']['lean_deg']))
        Tbb = np.eye(4); Tbb[:3, 3] = [.8, .15, .14]
        Tbc = np.eye(4); Tbc[:3, :3] = np.diag([1., -1., -1.]); Tbc[:3, 3] = [.8, .15, .60]
        root, tip = np.array([.08, -.10, root_z]), np.array([.055, -.10, tip_z])
        uv = self.camera._pin_project(np.stack([root, tip]) @ Tbb[:3, :3].T + Tbb[:3, 3],
                                      np.linalg.inv(Tbc), self.K)[0]
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp)/'sample.raw.png'
            import cv2
            cv2.imwrite(str(photo), np.full((240, 320, 3), 230, np.uint8))
            photo.with_name('sample.json').write_text(json.dumps(dict(
                T_base_basket=Tbb.tolist(), T_base_flange=Tbc.tolist(),
                T_flange_cam=np.eye(4).tolist(), K=self.K.tolist())))
            labels = Path(tmp)/'labels.json'
            labels.write_text(json.dumps(dict(photos=[dict(image=str(photo), pins=[dict(
                row='pair_A_row_plus', root_uv=uv[0].tolist(), tip_uv=uv[1].tolist(), review='checked')])])))
            result = self.camera.pin_layout_measure(labels, Path(tmp)/'out')
            measured = result['pins'][0]
            np.testing.assert_allclose(measured['root_at_rail_basket_xyz'], root, atol=1e-9)
            np.testing.assert_allclose(measured['top_basket_xyz'], tip, atol=1e-9)
            self.assertAlmostEqual(measured['base_basket_xyz'][2], floor)


class ModelOverlayTests(unittest.TestCase):
    def test_pin_quadrant_uses_centre_clip_boundary(self):
        import camera_tools as camera
        clip_x = 0.035
        self.assertEqual(camera._pin_quadrant(0.020, -0.110, clip_x), 'Q4')
        self.assertEqual(camera._pin_quadrant(0.050, -0.110, clip_x), 'Q1')
        self.assertEqual(camera._pin_quadrant(0.020, 0.129, clip_x), 'Q3')
        self.assertEqual(camera._pin_quadrant(0.050, 0.129, clip_x), 'Q2')

    def test_partial_photo_slot_number_keeps_global_model_id(self):
        import camera_tools as camera
        from plate_plans import basket_gaps, pin_gap

        gaps = basket_gaps()
        for yaw_deg in (0.1, 180.1):
            Tbb = np.eye(4)
            a = np.radians(yaw_deg)
            Tbb[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            Tbb[:3, 3] = [0.807, -0.192, 0.1442]
            for quadrant, slots in gaps.items():
                # Model only a cropped subset; global numbers must not restart at 1.
                target_numbers = sorted({1, max(2, len(slots) // 2), len(slots)})
                groups = [dict(quadrant=quadrant,
                               local_x_m=float(slots[pin_gap(Tbb, quadrant, n)][0]))
                          for n in target_numbers]
                camera._assign_gap_numbers(Tbb, groups, gaps)
                self.assertEqual([g['number'] for g in groups], target_numbers)
                self.assertEqual(len({g['number'] for g in groups}), len(groups))

    def test_competing_partial_slot_matches_remain_unaddressed(self):
        import camera_tools as camera
        from plate_plans import basket_gaps, pin_gap

        Tbb = np.eye(4)
        Tbb[:3, 3] = [0.807, -0.192, 0.1442]
        gaps = basket_gaps()
        quadrant, number = 'Q2', 6
        x = float(gaps[quadrant][pin_gap(Tbb, quadrant, number)][0])
        groups = [dict(quadrant=quadrant, local_x_m=x), dict(quadrant=quadrant, local_x_m=x)]
        camera._assign_gap_numbers(Tbb, groups, gaps)
        self.assertEqual([g['number'] for g in groups], [None, None])
        self.assertEqual([g['number_candidate'] for g in groups], [number, number])

    def test_tine_numbering_is_far_to_near_for_all_quadrants(self):
        """(10/15, Codex AA review) the overlay's own numbering function: 1 = farthest from the robot in every quadrant."""
        import camera_tools as camera
        cx = 0.035
        pins = [(np.array([cx + x, y, 0.0]), np.array([cx + x, y, 0.1]))
                for y in (-0.1, 0.1) for x in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)]
        for y in (-0.1, 0.1):
            self.assertEqual(camera.tine_number(pins, np.array([cx + 0.3, y, 0.0]), cx), 1)    # far rows: outer end = 1
            self.assertEqual(camera.tine_number(pins, np.array([cx + 0.1, y, 0.0]), cx), 3)    # next to the clip = last
            self.assertEqual(camera.tine_number(pins, np.array([cx - 0.1, y, 0.0]), cx), 1)    # near rows: next to the clip = 1
            self.assertEqual(camera.tine_number(pins, np.array([cx - 0.3, y, 0.0]), cx), 3)    # robot-side end = last

    def test_overlay_draws_model_tines(self):
        """(10/14) model_overlay: tines projected onto a saved wrist photo (camera 0.42 m above the basket, looking down)."""
        import cv2
        import tempfile
        import camera_tools as camera
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.imwrite(str(tmp / 'shot.raw.png'), img)
            Tbf = np.eye(4)
            Tbf[:3, :3] = np.diag([1.0, -1.0, -1.0])        # z down
            Tbf[:3, 3] = [-0.1, 0.1, 0.42]
            meta = dict(T_base_basket=np.eye(4).tolist(), T_base_flange=Tbf.tolist(), T_flange_cam=np.eye(4).tolist(),
                        K=[[400, 0, 320], [0, 400, 240], [0, 0, 1]], png=str(tmp / 'shot.png'))
            (tmp / 'shot.json').write_text(json.dumps(meta))
            (tmp / 'shot.req.json').write_text(json.dumps(meta))
            out = camera.model_overlay(tmp / 'shot.req.json', tmp / 'overlay.png')
            self.assertGreater(int((cv2.imread(str(out)) > 0).sum()), 1000)


class CameraPlaneTests(unittest.TestCase):
    def test_rgbd_plane_yaw_and_tilt_with_two_mm_noise(self):
        import camera_tools
        K = np.array([[600., 0, 160.], [0, 600., 120.], [0, 0, 1.]])
        # Camera optical +z looks along base +x, so a vertical dish faces it.
        Tbc = np.eye(4)
        Tbc[:3, :3] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        yy, xx = np.indices((240, 320))
        ray = np.stack(((xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)), axis=-1)
        rgb = np.zeros((240, 320, 3), np.uint8)
        rgb[40:200, 70:250] = (0, 0, 220)
        rng = np.random.default_rng(7)
        for yaw, tilt in ((0., 0.), (-10., 0.), (10., 0.), (0., 8.), (10., 8.)):
            yr, tr = np.radians([yaw, tilt])
            normal_base = np.array([np.cos(tr)*np.cos(yr), np.cos(tr)*np.sin(yr), np.sin(tr)])
            normal_cam = Tbc[:3, :3].T @ normal_base
            depth = (.30*normal_cam[2] / (ray @ normal_cam)) + rng.normal(0, .002, (240, 320))
            depth[::11, ::9] = 0  # missing stereo pixels
            result = camera_tools.plate_plane(rgb, depth, K, Tbc, np.eye(4))
            self.assertAlmostEqual(result['yaw_error_deg'], yaw, delta=1.)
            self.assertAlmostEqual(result['tilt_error_deg'], tilt, delta=1.)
            self.assertGreater(result['inlier_fraction'], .8)

    def test_rgbd_plane_rejects_missing_close_depth(self):
        import camera_tools
        rgb = np.zeros((240, 320, 3), np.uint8)
        rgb[40:200, 70:250] = (0, 0, 220)
        with self.assertRaisesRegex(ValueError, 'Insufficient red-plate depth'):
            camera_tools.plate_plane(rgb, np.zeros((240, 320)), np.eye(3), np.eye(4), np.eye(4))

    def test_fixed_camera_horizontal_dish_with_occlusion(self):
        import camera_tools
        K = np.array([[600., 0, 160.], [0, 600., 120.], [0, 0, 1.]])
        Tbc = np.diag([1., -1., -1., 1.])  # optical axis looks down
        yy, xx = np.indices((240, 320))
        radius = ((xx - 160.) / 85.) ** 2 + ((yy - 120.) / 70.) ** 2
        rgb = np.zeros((240, 320, 3), np.uint8)
        rgb[radius < 1.] = (0, 0, 220)
        rgb[85:155, 155:165] = 0  # gripper or basket occlusion
        depth = 1.20 + 0.006 * radius + np.random.default_rng(8).normal(0, .002, radius.shape)
        depth[::12, ::11] = 0
        result = camera_tools.plate_plane(rgb, depth, K, Tbc, np.eye(4),
                                          depth_range_m=(1.0, 1.4), surface_tolerance_m=.015)
        self.assertLess(result['tilt_from_horizontal_deg'], 1.)
        self.assertGreater(result['inlier_fraction'], .8)
        self.assertFalse(result['yaw_observable'])
        self.assertIsNone(result['yaw_error_deg'])


class BasketGapNumbering(unittest.TestCase):
    def test_pin_gap_maps_far_to_near_numbering(self):
        import plate_plans

        # New basket pose: basket +x points away from the robot.
        Tbb = make_T(R.from_euler('z', 0.1, degrees=True).as_matrix(),
                     [0.807, -0.192, 0.0])
        Tbb_reversed = make_T(R.from_euler('z', 180, degrees=True).as_matrix(),
                              [0.807, -0.192, 0.0])
        gaps = plate_plans.basket_gaps()
        self.assertEqual({q: len(gaps[q]) for q in gaps},
                         {'Q1': 6, 'Q2': 8, 'Q3': 12, 'Q4': 12})
        for q, qgaps in gaps.items():
            xs = [float(g[0]) for g in qgaps]
            self.assertTrue(all(a < b for a, b in zip(xs, xs[1:])), q)
            for pin in range(1, len(qgaps) + 1):
                # Number 1 is farthest; model gap arrays are robot-side -> far-side.
                self.assertEqual(plate_plans.pin_gap(Tbb, q, pin), len(qgaps) - pin,
                                 f'{q} pin {pin}')
                self.assertEqual(plate_plans.pin_gap(Tbb_reversed, q, pin), pin - 1,
                                 f'reversed {q} pin {pin}')
            with self.assertRaises(ValueError):
                plate_plans.pin_gap(Tbb, q, 0)
            with self.assertRaises(ValueError):
                plate_plans.pin_gap(Tbb, q, len(qgaps) + 1)


class SlotTargetGenerationTests(unittest.TestCase):
    def test_slot_targets_reproduce_reviewed_table_and_refuse_overwrite(self):
        import camera_tools

        reference_path = DATA / 'basket' / 'slot_targets_20261014_codex_v1.json'
        reference = json.loads(reference_path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'slot_targets.json'
            generated = camera_tools.slot_targets(output)
            # Row centers are rounded to 1 um; tolerate a one-unit decimal
            # tie difference between the serialized review JSON and refit.
            generated_cmp = json.loads(json.dumps(generated))
            reference_cmp = json.loads(json.dumps(reference))
            # Implementation hashes are provenance only. Planner/geometry code
            # can change while the reviewed input data and target table remain
            # identical; data-source hashes below still detect stale inputs.
            generated_cmp['sources']['pin_gap_implementation'].pop('sha256', None)
            reference_cmp['sources']['pin_gap_implementation'].pop('sha256', None)
            generated_cmp['sources']['basket_geometry'].pop('sha256', None)
            reference_cmp['sources']['basket_geometry'].pop('sha256', None)
            for actual, expected in zip(generated_cmp['targets'], reference_cmp['targets']):
                for arow, erow in zip(actual['rows_used'], expected['rows_used']):
                    axy = arow.pop('row_gap_centre_xy_m')
                    exy = erow.pop('row_gap_centre_xy_m')
                    np.testing.assert_allclose(axy, exy, atol=1.01e-6, rtol=0)
            self.assertEqual(generated_cmp, reference_cmp)
            with self.assertRaises(FileExistsError):
                camera_tools.slot_targets(output)



class DemoDataTests(unittest.TestCase):
    def test_full_11_to_14_only_known_contact_demo_is_unsafe(self):
        manifest = json.loads((demo_data.TEACH_DIR / "bc_demos.json").read_text())
        reports = [demo_data.demo_audit(demo_data.TEACH_DIR / d["file"]) for d in manifest["demos"]]
        by_name = {r["demo"]: r for r in reports}
        self.assertEqual(set(by_name), {"full_11.json", "full_12.json", "full_13.json", "full_14.json"})
        self.assertEqual([name for name, r in by_name.items() if not r["safe"]], ["full_14.json"])
        self.assertTrue(by_name["full_14.json"]["wrench"]["tool_force"]["events"])
        self.assertTrue(by_name["full_14.json"]["wrench"]["ext_torque"]["events"])
        self.assertTrue(by_name["full_14.json"]["collision"]["frame"]["violation_events"])
        self.assertLess(by_name["full_14.json"]["collision"]["frame"]["minimum_clearance_m"],
                        by_name["full_14.json"]["collision"]["frame"]["margin_m"])
        self.assertTrue(all("collision" in r and "speed" in r for r in reports))
        self.assertGreater(by_name["full_14.json"]["speed"]["max_joint_axis_deg_s"], 0)
        self.assertGreater(by_name["full_14.json"]["speed"]["max_tcp_speed_mm_s_fk"], 0)

    def test_keyposes_export_tcp_in_basket_frame(self):
        kin = Kinematics()
        q = np.array([-5.98, -14.75, 117.74, -3.92, 7.46, -85.07])
        flange = kin.fk(np.radians(q))
        posx = np.r_[flange[:3, 3] * 1000., [0., 0., 0.]]
        poses = {name: {"posj": q.tolist(), "posx": posx.tolist()} for name in demo_data.KEYPOSES}
        Tbb = make_T(R.from_euler("z", 0.3).as_matrix(), [0.9, 0.2, 0.15])
        result = demo_data.transform_keyposes(poses, Tbb, kin)
        expected_tcp = demo_data._tcp_from_flange(flange, demo_data.DEFAULT_TOOL_LENGTH_M)
        expected_basket = np.linalg.inv(Tbb) @ expected_tcp
        np.testing.assert_allclose(result["keypose_T_basket_tcp"][0], expected_basket, atol=1e-10)
        self.assertLess(float(result["keypose_fk_flange_xyz_error_mm"].max()), 1.0)

    def test_manifest_layout_and_unsafe_flags_exclude(self):
        old_layout = "old: basket y +0.17~+0.20, plate on the -y stand"
        common = {"layout": old_layout, "quadrant": "Q1", "landed_pin": 5}
        reasons = demo_data.default_exclusion_reasons(common, demo_data.DEFAULT_TARGET_LAYOUT)
        self.assertEqual(reasons, ["layout_mismatch_or_unknown"])
        reasons = demo_data.default_exclusion_reasons(common | {"unsafe": True}, old_layout)
        self.assertEqual(reasons, ["manifest_unsafe"])


class BasketReplayReportTests(unittest.TestCase):
    def test_replay_report_uses_stored_gates_and_refuses_overwrite(self):
        import goal1_offline

        payload = {
            "results": [
                {"run": "r1_Q1_pin1", "sim_rate": 0.0},
                {"run": "r2_Q1_pin2", "sim_rate": 0.5},
                {"run": "r3_Q2_pin1", "sim_rate": 0.0},
            ],
            "names": ["r1_Q1_pin1", "r1_Q1_pin1", "r2_Q1_pin2", "r2_Q1_pin2", "r3_Q2_pin1", "r3_Q2_pin1"],
            "final": {"low": [0.050, 0.070, 0.050, 0.050, 0.055, 0.065]},
            "quadrant_ok": [True, True, True, False, True, True],
            "in_quadrant": [True, True, True, True, True, False],
            "seated": [True, False, True, True, True, False],
            "landed_gap": [1, 1, 2, 2, 3, 2],
            "target_gap": [1, 1, 2, 2, 3, 2],
            "gap_err": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay_fixture.json"
            replay.write_text(json.dumps(payload))
            out = root / "report"
            result = goal1_offline.basket_replay_report([replay], out)
            q1 = next(row for row in result["rows"] if row["quadrant"] == "Q1")
            q2 = next(row for row in result["rows"] if row["quadrant"] == "Q2")
            self.assertEqual(q1["plans"], 2)
            self.assertEqual(q1["episodes"], 4)
            self.assertAlmostEqual(q1["in_quadrant_rate"], 1.0)
            self.assertAlmostEqual(q1["exact_rate"], 0.75)
            self.assertAlmostEqual(q1["seated_exact_rate"], 0.5)
            self.assertAlmostEqual(q1["strict_sim_rate"], 0.25)
            self.assertAlmostEqual(q2["seated_exact_rate"], 0.5)
            self.assertTrue(Path(result["markdown"]).is_file())
            self.assertTrue(Path(result["plot"]).is_file())
            report = Path(result["markdown"]).read_text()
            self.assertIn("50.0%", report)
            self.assertIn("2.50 / 3.70 / 4.00", report)
            with self.assertRaises(FileExistsError):
                goal1_offline.basket_replay_report([replay], out)


class BasketInsertRewardTests(unittest.TestCase):
    def test_optional_unseated_penalty_is_opt_in_and_only_applies_to_unseated(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('basket_train_reward_test', root / 'rl/basket/train.py')
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        exact = np.array([True, False, False, False])
        seated = np.array([True, True, False, False])
        in_quadrant = np.array([True, True, True, False])
        landed = np.array([2, 3, 2, -1])
        target = np.array([2, 2, 2, 2])

        legacy = train.insert_reward(exact, seated, in_quadrant, landed, target)
        np.testing.assert_allclose(legacy, [1.0, 0.15, 0.0, -0.2])
        changed = train.insert_reward(exact, seated, in_quadrant, landed, target, unseated_penalty=0.25)
        np.testing.assert_allclose(changed, [1.0, 0.15, -0.25, -0.45])


class BasketGeometryClearanceTests(unittest.TestCase):
    def test_vertical_descent_capsule_contact_matches_analytic_cases(self):
        import goal1_offline

        point = np.array([[0.0, 0.0, 0.0]])
        vertical = goal1_offline._vertical_capsule_contact_ceiling(point, [0, 0, 0], [0, 0, 1], .01)
        horizontal = goal1_offline._vertical_capsule_contact_ceiling(point, [-1, 0, .5], [1, 0, .5], .01)
        missing = goal1_offline._vertical_capsule_contact_ceiling(point, [-1, .02, .5], [1, .02, .5], .01)
        self.assertAlmostEqual(vertical, 1.01, places=9)
        self.assertAlmostEqual(horizontal, .51, places=9)
        self.assertIsNone(missing)


class BasketG0ActionTests(unittest.TestCase):
    @staticmethod
    def expert_module():
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('basket_expert_g0_test', root / 'rl/basket/expert.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_recorded_tcp_delta_is_transformed_to_slot_frame_and_kept_under_caps(self):
        mod = self.expert_module()
        trace = dict(
            obs=[np.zeros((1, 17), np.float32)],
            pos=[np.array([[0.0, 0.0, 0.0]])],
            quat=[np.array([[1.0, 0.0, 0.0, 0.0]])],
            next_pos=[np.array([[0.0, 0.001, 0.0]])],
            next_quat=[np.array([[1.0, 0.0, 0.0, 0.0]])],
            open=[np.array([1.0])],
        )
        obs, act, clipped, frames = mod.g0_actions_from_trace(
            trace, [np.pi / 2], [0.00167, 0.00167, 0.00167, 0.0058, 0.0058, 0.0058])
        self.assertEqual(obs.shape, (1, 1, 17))
        self.assertEqual(act.shape, (1, 1, 7))
        self.assertAlmostEqual(float(act[0, 0, 0]), 0.001, places=6)
        self.assertAlmostEqual(float(act[0, 0, 1]), 0.0, places=6)
        self.assertEqual(float(act[0, 0, 6]), 1.0)
        self.assertEqual((clipped, frames), (0, 1))

    def test_tracking_action_uses_live_student_pose_and_caps_the_correction(self):
        mod = self.expert_module()
        action, clipped = mod.g0_tracking_action(
            [[-0.001, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]],
            [[0.002, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [0.0],
            [0.00167] * 3 + [0.0058] * 3)
        self.assertAlmostEqual(float(np.linalg.norm(action[0, :3])), 0.00167, places=7)
        self.assertTrue(clipped[0])
        from scipy.spatial.transform import Rotation
        target_q_xyzw = Rotation.from_rotvec([0.0, 0.0, 0.01]).as_quat()
        action, clipped = mod.g0_tracking_action(
            [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            [[target_q_xyzw[3], *target_q_xyzw[:3]]], [0.0], [0.00167] * 3 + [0.0058] * 3)
        self.assertAlmostEqual(float(np.linalg.norm(action[0, 3:6])), 0.0058, places=7)
        self.assertTrue(clipped[0])

    def test_joint_reference_actions_follow_targets_and_respect_per_frame_speed_caps(self):
        mod = self.expert_module()
        trace = dict(
            joint_target_deg=[np.array([[1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                              np.array([[1.5, -2.2, 0.0, 0.0, 0.0, 0.0],
                                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])],
            joint_speed_deg_s=[np.array([30.0, 6.0]), np.array([15.0, 6.0])],
            open=[np.array([0.0, 0.0]), np.array([1.0, 0.0])])
        target, commanded, action, clipped, frames = mod.g0_joint_actions_from_trace(
            trace, [[0.0] * 6, [0.0] * 6], hz=30.0)
        self.assertEqual(target.shape, (2, 2, 6))
        self.assertEqual(commanded.shape, (2, 2, 6))
        self.assertEqual(action.shape, (2, 2, 7))
        self.assertEqual((clipped, frames), (2, 4))
        np.testing.assert_allclose(action[0, :, 0], [1.0, 0.5])
        np.testing.assert_allclose(action[0, :, 1], [-1.0, -0.5])
        np.testing.assert_allclose(commanded[0], [[1.0, -1.0, 0, 0, 0, 0],
                                                 [1.5, -1.5, 0, 0, 0, 0]])
        command_delta = np.diff(np.concatenate([np.zeros((2, 1, 6)), commanded], axis=1), axis=1)
        per_frame_cap = np.array([[30.0, 15.0], [6.0, 6.0]])[:, :, None] / 30.0
        np.testing.assert_allclose(action[:, :, :6], command_delta)
        self.assertTrue((np.abs(action[:, :, :6]) <= per_frame_cap + 1e-7).all())
        np.testing.assert_array_equal(action[..., 6], [[0.0, 1.0], [0.0, 0.0]])
        previous_commanded = np.concatenate([np.zeros((2, 1, 6)), commanded[:, :-1]], axis=1)
        clipped_mask = mod.g0_joint_action_clipped_mask(target, previous_commanded, action[..., :6])
        self.assertEqual(int(clipped_mask.sum()), clipped)
        np.testing.assert_array_equal(clipped_mask, [[True, True], [False, False]])

    def test_joint_action_clip_mask_uses_previous_commanded_target(self):
        mod = self.expert_module()
        target = np.array([[[2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]]], dtype=float)
        previous = np.array([[[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]], dtype=float)
        action = np.array([[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]], dtype=float)
        # The second teacher target moves only 1 degree from the first teacher target,
        # but 2 degrees from the previous commanded target after the first clip.
        np.testing.assert_array_equal(
            mod.g0_joint_action_clipped_mask(target, previous, action), [[True, True]])

    def test_joint_reference_actions_reject_bad_shapes_nan_and_speed(self):
        mod = self.expert_module()
        base = dict(joint_target_deg=[np.zeros((1, 6))],
                    joint_speed_deg_s=[np.array([30.0])], open=[np.array([0.0])])
        with self.assertRaisesRegex(ValueError, 'shape'):
            mod.g0_joint_actions_from_trace(base, np.zeros((2, 6)))
        bad_nan = dict(base, joint_target_deg=[np.full((1, 6), np.nan)])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            mod.g0_joint_actions_from_trace(bad_nan, np.zeros((1, 6)))
        bad_speed = dict(base, joint_speed_deg_s=[np.array([0.0])])
        with self.assertRaisesRegex(ValueError, 'positive'):
            mod.g0_joint_actions_from_trace(bad_speed, np.zeros((1, 6)))
        with self.assertRaisesRegex(ValueError, 'positive'):
            mod.g0_joint_actions_from_trace(base, np.zeros((1, 6)), hz=0.0)
        bad_open = dict(base, open=[np.array([0.5])])
        with self.assertRaisesRegex(ValueError, 'binary'):
            mod.g0_joint_actions_from_trace(bad_open, np.zeros((1, 6)))

    def test_closed_loop_extra_accepts_only_declared_depth_sweep(self):
        mod = self.expert_module()
        for value in (0.005, 0.010, 0.015):
            self.assertAlmostEqual(mod.validate_closed_loop_extra(value), value)
        for value in (-0.005, 0.020, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                mod.validate_closed_loop_extra(value)

    def test_teacher_rate_cap_retimes_each_path_and_reaches_its_endpoint(self):
        mod = self.expert_module()
        initial = np.zeros((3, 6), dtype=float)
        paths = [np.array([[1.0, -2.0, 0, 0, 0, 0], [2.0, -2.0, 0, 0, 0, 0]]),
                 np.array([[0.0, 0.5, 0, 0, 0, 0]]), None]
        target = mod.teacher_rate_cap_targets(paths, initial, speed_deg_s=6.0, hz=30.0)
        self.assertEqual(target.shape[0], 3)
        self.assertGreater(target.shape[1], 1)
        delta = np.diff(np.concatenate([initial[:, None], target], axis=1), axis=1)
        self.assertLessEqual(float(np.max(np.abs(delta))), 6.0 / 30.0 + 1e-7)
        np.testing.assert_allclose(target[0, -1], paths[0][-1], atol=1e-10)
        np.testing.assert_allclose(target[1, -1], paths[1][-1], atol=1e-10)
        np.testing.assert_allclose(target[2], 0.0, atol=0.0)

    def test_teacher_rate_cap_phase_change_starts_from_prior_commanded_target(self):
        mod = self.expert_module()
        initial = np.zeros((1, 6), dtype=float)
        first = mod.teacher_rate_cap_targets([np.array([[2.0, 0, 0, 0, 0, 0]])], initial, 20.0, 30.0)
        second = mod.teacher_rate_cap_targets([np.array([[1.0, 0, 0, 0, 0, 0]])], first[:, -1], 1.0, 30.0)
        delta = np.diff(np.concatenate([first[:, -1:, :], second], axis=1), axis=1)
        self.assertLessEqual(float(np.max(np.abs(delta))), 1.0 / 30.0 + 1e-7)
        np.testing.assert_allclose(second[0, -1], [1, 0, 0, 0, 0, 0], atol=1e-10)

    def test_teacher_rate_cap_rejects_invalid_inputs(self):
        mod = self.expert_module()
        base = np.zeros((1, 6), dtype=float)
        with self.assertRaisesRegex(ValueError, 'initial joint'):
            mod.teacher_rate_cap_targets([None], np.zeros((1, 5)), 1.0)
        with self.assertRaisesRegex(ValueError, 'path count'):
            mod.teacher_rate_cap_targets([], base, 1.0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            mod.teacher_rate_cap_targets([np.full((1, 6), np.nan)], base, 1.0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            mod.teacher_rate_cap_targets([None], base, 0.0)

    def test_g0_contact_frames_use_filtered_force_threshold_and_none_for_no_contact(self):
        mod = self.expert_module()
        forces = np.array([[0.0, 0.9, 1.0, 0.2], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(mod.g0_first_contact_frames(forces, 1.0), [2, 0, None])
        with self.assertRaisesRegex(ValueError, 'shape'):
            mod.g0_first_contact_frames([0.0, 1.2], 1.0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            mod.g0_first_contact_frames([[0.0, np.nan]], 1.0)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            mod.g0_first_contact_frames([[0.0, -0.1]], 1.0)

    def test_g0_insert_telemetry_packs_episode_major_finite_frames(self):
        mod = self.expert_module()
        samples = [
            dict(low_m=np.array([.08, .07]), d_bottom_m=np.array([.01, .02]),
                 normal_err_deg=np.array([3.0, 4.0]), contact_force_n=np.array([0.0, 1.1])),
            dict(low_m=np.array([.05, .04]), d_bottom_m=np.array([.03, .04]),
                 normal_err_deg=np.array([5.0, 6.0]), contact_force_n=np.array([2.0, 0.0])),
        ]
        packed = mod.g0_stack_insert_telemetry(samples, 2, 1.0)
        self.assertEqual(packed['low_m'].shape, (2, 2))
        np.testing.assert_allclose(packed['low_m'], [[.08, .05], [.07, .04]])
        np.testing.assert_array_equal(packed['contact'], [[False, True], [True, False]])
        self.assertEqual(packed['first_contact_frame'], [1, 0])
        empty = mod.g0_stack_insert_telemetry([], 2, 1.0)
        self.assertEqual(empty['d_bottom_m'].shape, (2, 0))
        self.assertEqual(empty['first_contact_frame'], [None, None])
        bad = [dict(sample, normal_err_deg=np.array([np.nan, 0.0])) for sample in samples[:1]]
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            mod.g0_stack_insert_telemetry(bad, 2)

    def test_rotation_hold_composes_spatial_delta_and_is_opt_in(self):
        import torch
        from rl_utils import ROTATION_HOLD_DEFAULT, integrate_spatial_quat
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, np.deg2rad(10.0)]])
        q1 = integrate_spatial_quat(q, d)
        q2 = integrate_spatial_quat(q1, d)
        expected = torch.tensor([[np.cos(np.deg2rad(10.0)), 0.0, 0.0, np.sin(np.deg2rad(10.0))]], dtype=torch.float32)
        torch.testing.assert_close(q2, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(ROTATION_HOLD_DEFAULT)
        from scipy.spatial.transform import Rotation
        qx = Rotation.from_euler('x', 30, degrees=True).as_quat()
        qx_wxyz = torch.tensor([[qx[3], qx[0], qx[1], qx[2]]], dtype=torch.float32)
        dy = torch.tensor([[0.0, np.deg2rad(15.0), 0.0]], dtype=torch.float32)
        qxy = integrate_spatial_quat(qx_wxyz, dy).numpy()[0]
        expected_xyzw = (Rotation.from_rotvec([0.0, np.deg2rad(15.0), 0.0]) * Rotation.from_quat(qx)).as_quat()
        np.testing.assert_allclose(qxy, expected_xyzw[[3, 0, 1, 2]], atol=1e-6)

    def test_g0_seated_exact_metric_does_not_depend_on_finger_rim_diagnostic(self):
        mod = self.expert_module()
        np.testing.assert_array_equal(
            mod.g0_success_mask([True, True, False], [True, True, True], [True, False, True]),
            [True, False, False])

    def test_g0_zero_teacher_denominator_is_not_reported_as_a_failed_rate(self):
        mod = self.expert_module()
        self.assertEqual(mod.g0_gate_status(0, 0), ('insufficient_teacher_successes', None))
        self.assertEqual(mod.g0_gate_status(20, 19), ('pass', 0.95))
        self.assertEqual(mod.g0_gate_status(20, 18), ('fail', 0.9))
        with self.assertRaises(ValueError):
            mod.g0_gate_status(1, 2)

    def test_g0_timeout_is_diagnostic_and_cursor_can_continue(self):
        mod = self.expert_module()
        latched, expired = mod.g0_gate_timeout_state(
            [True, True, False], [False, False, False], [59, 60, 60], 60)
        np.testing.assert_array_equal(expired, [False, True, False])
        np.testing.assert_array_equal(latched, [False, True, False])
        self.assertTrue(mod.g0_replay_cursor_should_advance(True, [True, True, False],
                                                            [False, False, False], [False, False, False]))
        self.assertFalse(mod.g0_replay_cursor_should_advance(False, [True, True], [True, False], None))
        self.assertTrue(mod.g0_replay_cursor_should_advance(False, [True, True], [False, False], None))
        self.assertFalse(mod.g0_replay_cursor_should_advance(False, [True], [True], None, [False]))
        self.assertTrue(mod.g0_replay_cursor_should_advance(False, [True], [True], None, [True]))
        with self.assertRaises(ValueError):
            mod.g0_gate_timeout_state([True], [False, False], [1], 60)

    def test_g0_sparse_phase_gates_and_strict_alignment_tolerances(self):
        mod = self.expert_module()
        phases = ['preinsert', 'reaim_0', 'reaim_0', 'reaim_1', 'insert', 'insert', 'release']
        np.testing.assert_array_equal(
            mod.g0_phase_gate_mask(phases),
            [False, False, True, True, False, True, False])
        self.assertTrue(mod.g0_gate_alignment_mask([0.000999], [np.deg2rad(0.499)])[0])
        self.assertFalse(mod.g0_gate_alignment_mask([0.001], [0.0])[0])
        self.assertFalse(mod.g0_gate_alignment_mask([0.0], [np.deg2rad(0.5)])[0])
        with self.assertRaises(ValueError):
            mod.g0_phase_gate_mask([])
        with self.assertRaises(ValueError):
            mod.g0_gate_alignment_mask([np.nan], [0.0])

    def test_g0_lift_requires_the_existing_physical_finger_and_rim_checks(self):
        mod = self.expert_module()
        # Match plate_env._state: stalled fingers plus TCP on the plate rim; a 5 cm lift alone is insufficient.
        tcp_local = np.array([[0.128, 0.0, 0.016], [0.128, 0.0, 0.016],
                              [0.128, 0.0, 0.016], [0.0, 0.0, 0.016]])
        result = mod.g0_physical_grasp([0.7, 0.2, 0.7, 0.7], tcp_local,
                                       [True, True, False, True], [0.5, 0.97], 0.128, 0.016, 0.03)
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_g0_requires_physical_grasp_until_the_first_release_action(self):
        mod = self.expert_module()
        held = [np.array([True, True]), np.array([True, False]), np.array([False, False])]
        released = [np.array([False, False]), np.array([False, False]), np.array([True, True])]
        np.testing.assert_array_equal(mod.g0_trace_grasp_maintained(held, released), [True, False])

    def test_recorded_diagonal_action_respects_vector_speed_caps(self):
        from scipy.spatial.transform import Rotation
        mod = self.expert_module()
        q_xyzw = Rotation.from_rotvec([0.005, 0.005, 0.0]).as_quat()
        trace = dict(
            obs=[np.zeros((1, 17), np.float32)],
            pos=[np.zeros((1, 3))], quat=[np.array([[1.0, 0.0, 0.0, 0.0]])],
            next_pos=[np.array([[0.002, 0.002, 0.0]])],
            next_quat=[np.array([[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]])],
            open=[np.array([0.0])])
        _, act, clipped, frames = mod.g0_actions_from_trace(
            trace, [0.0], [0.00167] * 3 + [0.0058] * 3)
        self.assertAlmostEqual(float(np.linalg.norm(act[0, 0, :3])), 0.00167, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(act[0, 0, 3:6])), 0.0058, places=7)
        self.assertTrue((np.abs(act[0, 0, :6]) <= np.array([0.00167] * 3 + [0.0058] * 3)).all())
        self.assertEqual((clipped, frames), (1, 1))

    def test_rotation_trace_supports_multiple_envs_and_frames(self):
        from scipy.spatial.transform import Rotation
        mod = self.expert_module()
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        rz = Rotation.from_rotvec([0.0, 0.0, 0.002]).as_quat()
        quat_xyzw = np.array([rz[3], rz[0], rz[1], rz[2]])
        trace = dict(
            obs=[np.zeros((2, 17), np.float32) for _ in range(2)],
            pos=[np.zeros((2, 3)) for _ in range(2)],
            quat=[np.tile(identity, (2, 1)) for _ in range(2)],
            next_pos=[np.zeros((2, 3)) for _ in range(2)],
            next_quat=[np.tile(quat_xyzw, (2, 1)) for _ in range(2)],
            open=[np.array([0.0, 1.0]), np.array([1.0, 0.0])])
        obs, act, clipped, frames = mod.g0_actions_from_trace(
            trace, [0.0, 0.2], [0.00167] * 3 + [0.0058] * 3)
        self.assertEqual(obs.shape, (2, 2, 17))
        self.assertEqual(act.shape, (2, 2, 7))
        np.testing.assert_allclose(act[..., 5], 0.002, atol=1e-7)
        np.testing.assert_array_equal(act[..., 6], [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual((clipped, frames), (0, 4))

    def test_g0_refuses_nonfinite_or_misaligned_trace(self):
        mod = self.expert_module()
        base = dict(obs=[np.zeros((1, 17), np.float32)], pos=[np.zeros((1, 3))],
                    quat=[np.array([[1.0, 0.0, 0.0, 0.0]])], next_pos=[np.zeros((1, 3))],
                    next_quat=[np.array([[1.0, 0.0, 0.0, 0.0]])], open=[np.array([0.0])])
        bad_nan = dict(base, obs=[np.full((1, 17), np.nan, np.float32)])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            mod.g0_actions_from_trace(bad_nan, [0.0], [0.00167] * 3 + [0.0058] * 3)
        bad_length = dict(base, next_pos=[])
        with self.assertRaisesRegex(ValueError, 'same nonzero frame count'):
            mod.g0_actions_from_trace(bad_length, [0.0], [0.00167] * 3 + [0.0058] * 3)

    def test_g0_pose_error_reports_translation_and_quaternion_angle(self):
        from scipy.spatial.transform import Rotation
        mod = self.expert_module()
        q = Rotation.from_euler('z', 4, degrees=True).as_quat()[[3, 0, 1, 2]]
        dp, dr = mod.g0_pose_error([[0, 0, 0]], [[1, 0, 0, 0]], [[.003, 0, 0]], [q])
        np.testing.assert_allclose(dp, [.003])
        np.testing.assert_allclose(np.degrees(dr), [4.0], atol=1e-6)

    def test_g0_first_true_handles_per_episode_open_timing(self):
        mod = self.expert_module()
        self.assertEqual(mod.g0_first_true([[False, True, True], [False, False, True], [False, False, False]]),
                         [1, 2, None])

if __name__ == '__main__':
    unittest.main()


class BasketAlignmentTests(unittest.TestCase):
    """(10/15) rl/basket/env.rim_row_crossings: the stage-4 alignment observation geometry."""

    def test_upright_and_shifted_plate(self):
        import importlib.util
        import math
        import torch
        spec = importlib.util.spec_from_file_location('basket_obs', str(Path(__file__).resolve().parent.parent / 'rl/basket/env.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        r, rows = 0.134, [-0.0475, 0.0475]
        c = torch.tensor([[0.0, 0.0, 0.04 + r], [0.005, 0.01, 0.04 + r], [0.0, 0.2, 0.04 + r]])
        n = torch.tensor([[1.0, 0.0, 0.0]] * 3)
        x, z, ok = m.rim_row_crossings(c, n, rows, r)
        self.assertTrue(torch.allclose(x[0], torch.zeros(2), atol=1e-6))
        self.assertAlmostEqual(float(z[0, 0]), 0.04 + r - math.sqrt(r * r - 0.0475 ** 2), places=6)
        self.assertTrue(torch.allclose(x[1], torch.full((2,), 0.005), atol=1e-6))
        self.assertTrue(bool(ok[0].all()) and bool(ok[1].all()))
        self.assertFalse(bool(ok[2, 0]))                    # rim 20 cm across does not reach the far row
