"""Small regression checks for the offline candidate acceptance gate."""
import unittest

import numpy as np

from trajopt_backend import validate


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


if __name__ == "__main__":
    unittest.main()
