"""Focused hardware-free regression tests for demo_dataset.py."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import demo_dataset
from calib_core import make_T
from robot_model import Kinematics


class DemoDatasetTests(unittest.TestCase):
    def test_keyposes_export_tcp_in_basket_frame(self):
        kin = Kinematics()
        q = np.array([-5.98, -14.75, 117.74, -3.92, 7.46, -85.07])
        flange = kin.fk(np.radians(q))
        posx = np.r_[flange[:3, 3] * 1000., [0., 0., 0.]]
        poses = {name: {"posj": q.tolist(), "posx": posx.tolist()} for name in demo_dataset.KEYPOSES}
        Tbb = make_T(Rotation.from_euler("z", 0.3).as_matrix(), [0.9, 0.2, 0.15])
        result = demo_dataset.transform_keyposes(poses, Tbb, kin)
        expected_tcp = demo_dataset._tcp_from_flange(flange, demo_dataset.DEFAULT_TOOL_LENGTH_M)
        expected_basket = np.linalg.inv(Tbb) @ expected_tcp
        np.testing.assert_allclose(result["keypose_T_basket_tcp"][0], expected_basket, atol=1e-10)
        self.assertLess(float(result["keypose_fk_flange_xyz_error_mm"].max()), 1.0)

    def test_manifest_layout_and_unsafe_flags_exclude(self):
        old_layout = "old: basket y +0.17~+0.20, plate on the -y stand"
        common = {"layout": old_layout, "quadrant": "Q1", "landed_pin": 5}
        reasons = demo_dataset.default_exclusion_reasons(common, demo_dataset.DEFAULT_TARGET_LAYOUT)
        self.assertEqual(reasons, ["layout_mismatch_or_unknown"])
        reasons = demo_dataset.default_exclusion_reasons(common | {"unsafe": True}, old_layout)
        self.assertEqual(reasons, ["manifest_unsafe"])


if __name__ == "__main__":
    unittest.main()
