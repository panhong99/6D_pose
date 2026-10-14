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
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--offline-test', '--execute'])
        plan = dict(dry_run=True, accepted=True, offline_test=True, execution_allowed=False, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    margin_m=.1, legs=dict(approach=dict(waypoints_deg=[[0] * 6, [10] * 6], min_clearance_straight_m=.12)))
        self.assertTrue(nodes.validate(nodes.mover_args(['--plan', 'unused', '--leg', 'approach']), plan)[1])
        for option in ('--vel_deg_s', '--acc_deg_s2'):
            for value in ('-1', 'nan'):                    # no upper limit since 2026-10-13 (boss); only > 0, finite
                a = nodes.mover_args(['--plan', 'unused', '--leg', 'approach', option, value])
                self.assertTrue(nodes.validate(a, dict(plan, offline_test=False, execution_allowed=True))[1])

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

    def test_blank_floor_holes_not_pin(self):
        import cv2
        image = np.full((240, 320, 3), 230, np.uint8)
        for y in range(128, 176, 10):
            cv2.circle(image, (190, y), 2, (20, 20, 20), -1)
        self.assertNotEqual(self.match(image)['status'], 'detected')

    def test_plate_occlusion_is_not_a_measurement(self):
        image = self.fixture()
        image[:, 180:210] = (30, 30, 190)
        r = self.match(image)
        self.assertEqual(r['status'], 'occluded')
        self.assertIsNone(r['correction_mm'])

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

if __name__ == '__main__':
    unittest.main()
