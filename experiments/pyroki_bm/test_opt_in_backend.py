"""Contract checks for the offline opt-in wrapper."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from opt_in_backend import OfflinePyrokiBackend
from trajopt_backend import CandidateRejected


class Checker:
    def __init__(self, clearance=1.5):
        self.value = clearance

    def clearance(self, _):
        return self.value


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = OfflinePyrokiBackend.__new__(OfflinePyrokiBackend)
        joints = SimpleNamespace(lower_limits=np.full(6, -4.),
                                 upper_limits=np.full(6, 4.))
        self.backend.kinematics = SimpleNamespace(robot=SimpleNamespace(joints=joints))
        self.backend.robot_collision = object()
        self.start = np.zeros(6)
        self.goal = np.full(6, .1)
        self.kw = dict(held=Checker(), floor=Checker(), attached_cloud=np.zeros((1, 3)),
                       boxes=(), basket_T=np.eye(4), basket_half_xy=(.1, .1), floor_z=.2)

    def test_checked_candidate_returned(self):
        path = np.linspace(self.start, self.goal, 20)
        with patch('opt_in_backend.optimize', return_value=(path, [{'valid': True}])):
            out, attempts = self.backend.carry(self.start, self.goal, **self.kw)
        np.testing.assert_allclose(out, path)
        self.assertEqual(attempts, [{'valid': True}])

    def test_endpoint_mutation_rejected(self):
        path = np.linspace(self.start, self.goal, 20)
        path[-1, 0] += 1e-7
        with patch('opt_in_backend.optimize', return_value=(path, [{'seed': 'straight'}])):
            with self.assertRaises(CandidateRejected) as rejected:
                self.backend.carry(self.start, self.goal, **self.kw)
        self.assertEqual(rejected.exception.attempts[0]['seed'], 'straight')
        self.assertEqual(rejected.exception.attempts[-1]['reason'], 'endpoints_or_shape')

    def test_missing_checker_rejected(self):
        self.kw['floor'] = None
        with self.assertRaisesRegex(ValueError, 'required'):
            self.backend.carry(self.start, self.goal, **self.kw)

    def test_returned_candidate_is_rechecked(self):
        self.kw['floor'] = Checker(.5)
        path = np.linspace(self.start, self.goal, 20)
        with patch('opt_in_backend.optimize', return_value=(path, [{'seed': 'straight'}])):
            with self.assertRaises(CandidateRejected) as rejected:
                self.backend.carry(self.start, self.goal, **self.kw)
        self.assertEqual(rejected.exception.attempts[0]['seed'], 'straight')
        self.assertEqual(rejected.exception.attempts[-1]['reason'], 'final_exact_gate')

    def test_fk_ik_delegate_to_pyroki_bridge(self):
        self.backend.kinematics.fk = Mock(return_value=np.eye(4))
        self.backend.kinematics.ik = Mock(return_value=self.goal)
        np.testing.assert_array_equal(self.backend.fk(self.start), np.eye(4))
        np.testing.assert_array_equal(self.backend.ik(np.eye(4), self.start), self.goal)
        self.backend.kinematics.fk.assert_called_once_with(self.start)
        self.backend.kinematics.ik.assert_called_once()


if __name__ == '__main__':
    unittest.main()
