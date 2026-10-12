"""Offline tests of the real-robot pipeline (no camera, no robot, no ROS, no Isaac).

    cd calibration && python -m unittest test_core
"""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import real_pose_dryrun as dry
import ros2_pendant_mover as mover
import workcell as wc
from calib_utils import evaluate, inv_T, make_T, posx_to_T, solve_eye_to_hand, T_to_posx
from kinematics import DEFAULT_URDF, Kinematics, cube_yaw_deg, grasp_frames
from trajectory import polyline_path, smooth_path, time_parameterize

CELL = Path(__file__).resolve().parent / 'data' / 'real_cell_measured.json'


class Kinematic(unittest.TestCase):
    def setUp(self):
        self.kin = Kinematics(DEFAULT_URDF)

    def test_zero_pose_matches_the_real_robot(self):
        # read from the controller at q = 0: flange (0.02, 34.5, 1452.5) mm
        flange = self.kin.fk(np.zeros(6))[:3, 3] * 1000
        self.assertLess(np.abs(flange - [0.02, 34.5, 1452.5]).max(), 0.5)

    def test_ik_round_trip(self):
        q = np.radians([10, 40, 60, 5, 50, -20])
        got = self.kin.ik(self.kin.fk(q), q + 0.05)
        self.assertLess(np.abs(self.kin.fk(got) - self.kin.fk(q)).max(), 1e-4)

    def test_top_down_grasp_frame(self):
        for yaw in (0.0, 30.0, -20.0):
            cube = np.array([1.0, 0.05, 0.0285])
            T = make_T(np.eye(3), cube)
            T[:3, :3] = __import__('scipy.spatial.transform', fromlist=['Rotation']).Rotation.from_euler(
                'z', yaw, degrees=True).as_matrix()
            measured, tilt = cube_yaw_deg(T)
            self.assertAlmostEqual(measured, yaw, places=3)
            _, pre, _, q, _ = grasp_frames(self.kin, cube, measured, 0.0, 0.10)
            self.assertAlmostEqual(pre[2, 3], 0.1285, places=4)
            self.assertAlmostEqual(self.kin.fk(q)[2, 3], pre[2, 3], places=3)
            self.assertLess(abs(np.degrees(q[3])), 1.0)          # j4 ~ 0: tool points down


class Transforms(unittest.TestCase):
    def test_posx_round_trip(self):
        p = [967.75, 90.98, 173.33, 4.66, 168.52, 1.57]
        self.assertLess(np.abs(np.array(T_to_posx(posx_to_T(p))) - p).max(), 1e-6)

    def test_handeye_recovers_a_known_camera(self):
        rng = np.random.default_rng(0)
        from scipy.spatial.transform import Rotation as R
        Tbc = make_T(R.from_euler('xyz', [-125, 5, 178], degrees=True).as_matrix(), [1.5, 0.4, 0.1])
        Tfm = make_T(R.from_euler('xyz', [10, -5, 30], degrees=True).as_matrix(), [0.02, -0.01, 0.03])
        Tbf, Tcm = [], []
        for _ in range(20):
            a = make_T((R.from_euler('xyz', [180, 0, 0], degrees=True) * R.from_rotvec(rng.normal(0, .5, 3))).as_matrix(),
                       [rng.uniform(.4, .8), rng.uniform(-.3, .3), rng.uniform(.3, .7)])
            Tbf.append(a)
            Tcm.append(inv_T(Tbc) @ a @ Tfm)
        est, _ = solve_eye_to_hand(Tbf, Tcm)
        d = inv_T(Tbc) @ est
        self.assertLess(np.linalg.norm(d[:3, 3]), 1e-4)


class Cell(unittest.TestCase):
    def test_measured_frame_layout(self):
        cell = dict(wc.FRAME_CELL)
        cell.update({k: v for k, v in json.loads(CELL.read_text()).items() if not k.startswith('_')})
        boxes = wc.frame_layout(cell)
        up = {n: c for n, c, _ in boxes if n.startswith('upright')}
        self.assertEqual(sorted(round(v[0], 2) for v in up.values()), [0.85, 0.85, 1.65, 1.65])
        self.assertEqual(sorted(round(v[1], 2) for v in up.values()), [-0.56, -0.56, 0.56, 0.56])

    def test_plan_for_a_cube_in_front_of_the_robot(self):
        args = dry.parse_args(['--cell_json', str(CELL)])
        kin = Kinematics(DEFAULT_URDF)
        checker, _ = dry.build_checker(kin, args, make_T(np.eye(3), [1.5, 0.38, 0.115]))
        res = dry.plan_dryrun(kin, checker, make_T(np.eye(3), [1.04, 0.05, 0.0285]), np.zeros(6), args)
        self.assertTrue(res['accepted'], res['reasons'])
        for leg in res['legs'].values():
            self.assertGreaterEqual(leg['min_clearance_straight_m'], checker.margin)
            wp = np.array(leg['waypoints_deg'])
            self.assertLessEqual(np.abs(np.diff(wp, axis=0)).max(), args.max_segment_deg + 1e-6)

    def test_object_inside_the_frame_post_is_refused(self):
        args = dry.parse_args(['--cell_json', str(CELL)])
        kin = Kinematics(DEFAULT_URDF)
        checker, _ = dry.build_checker(kin, args, make_T(np.eye(3), [1.5, 0.38, 0.115]))
        res = dry.plan_dryrun(kin, checker, make_T(np.eye(3), [0.85, 0.50, 0.0285]), np.zeros(6), args)
        self.assertFalse(res['accepted'])


class Mover(unittest.TestCase):
    def plan(self, **kw):
        wp = [[0] * 6, [10, 10, 10, 0, 10, 0], [20, 20, 20, 0, 20, 0]]
        plan = dict(dry_run=True, accepted=True, margin_m=0.10, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    legs=dict(approach=dict(waypoints_deg=wp, min_clearance_straight_m=0.12)))
        plan.update(kw)
        return plan

    def check(self, plan, *argv):
        return mover.validate(mover.parse_args(['--plan', 'x', '--leg', 'approach', *argv]), plan)[1]

    def test_accepts_a_good_slow_plan(self):
        self.assertEqual(self.check(self.plan()), [])

    def test_refuses_fast_unaccepted_stale_or_big_moves(self):
        self.assertTrue(self.check(self.plan(), '--vel_deg_s', '10'))
        self.assertTrue(self.check(self.plan(accepted=False)))
        self.assertTrue(self.check(self.plan(created='2020-01-01 00:00:00')))
        big = self.plan()
        big['legs']['approach']['waypoints_deg'][1] = [80, 0, 0, 0, 0, 0]
        self.assertTrue(self.check(big))
        limit = self.plan()
        limit['legs']['approach']['waypoints_deg'][2] = [0, 0, 0, 0, 0, 359]
        self.assertTrue(self.check(limit, '--max_step_deg', '400'))

    def test_default_speed_is_the_slowest_setting(self):
        a = mover.parse_args(['--plan', 'x', '--leg', 'approach'])
        self.assertLessEqual(a.vel_deg_s, 2.0)
        self.assertLessEqual(a.acc_deg_s2, 2.0)


class Trajectory(unittest.TestCase):
    def test_smooth_path_timing_respects_limits(self):
        wp = np.radians([[0] * 6, [20, 10, 30, 0, 20, 0], [10, 40, 20, 0, 10, 5]])
        frames = time_parameterize(smooth_path(wp), np.radians(2), np.radians(2), 1 / 30)
        speed = np.abs(np.diff(frames, axis=0)).max() * 30
        self.assertLessEqual(np.degrees(speed), 2.0 * 1.05)
        self.assertTrue(np.allclose(frames[0], wp[0]) and np.allclose(frames[-1], wp[-1]))


if __name__ == '__main__':
    unittest.main()
