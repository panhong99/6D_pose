"""Small regression checks for the offline candidate acceptance gate."""
import unittest

import numpy as np

from trajopt_backend import _resample_path, validate
from benchmark_saved import insertion_gate


class Checker:
    def __init__(self, minimum):
        self.minimum = minimum

    def clearance(self, _):
        return self.minimum


class GateTests(unittest.TestCase):
    def setUp(self):
        self.path = np.array([[0., 0., 0., 0., 0., 0.],
                              [.1, 0., 0., 0., 0., 0.]])

    def test_clear_path_passes(self):
        self.assertTrue(validate(self.path, Checker(1.1), Checker(1.1), self.path[0])[0])

    def test_held_plate_collision_rejected(self):
        self.assertFalse(validate(self.path, Checker(.9), Checker(1.1), self.path[0])[0])

    def test_floor_violation_rejected(self):
        self.assertFalse(validate(self.path, Checker(1.1), Checker(.9), self.path[0])[0])

    def test_wrist_excursion_rejected(self):
        p = self.path.copy()
        p[-1, 3] = np.deg2rad(101)
        self.assertEqual(validate(p, Checker(1.1), Checker(1.1), p[0])[1]["reason"], "wrist")

    def test_limits_rejected(self):
        self.assertEqual(validate(self.path, Checker(1.1), Checker(1.1), self.path[0],
                                  limits=(np.full(6, -.05), np.full(6, .05)))[1]["reason"], "joint_limits")

    def test_external_seed_resampling_preserves_joint_endpoints(self):
        original = np.array([[0., 0., 0., 0., 0., 0.],
                             [.1, .2, 0., 0., 0., 0.],
                             [.3, .2, 0., 0., 0., 0.]])
        sampled = _resample_path(original, 20)
        self.assertEqual(sampled.shape, (20, 6))
        np.testing.assert_allclose(sampled[0], original[0])
        np.testing.assert_allclose(sampled[-1], original[-1])

    def test_shared_drop_wrist_is_diagnostic_only(self):
        drop = self.path.copy()
        drop[-1, 3] = np.deg2rad(110)
        result = insertion_gate(drop, Checker(1.1),
                                (np.full(6, -4.), np.full(6, 4.)), drop[0])
        self.assertTrue(result['valid'])
        self.assertFalse(result['wrist_limit_applies'])
        self.assertGreater(result['max_wrist_deg'], 100)


if __name__ == "__main__":
    unittest.main()
