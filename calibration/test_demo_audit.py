import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_audit import TEACH_DIR, demo_audit


class DemoAuditIntegrationTest(unittest.TestCase):
    def test_full_11_to_14_only_known_contact_demo_is_unsafe(self):
        manifest = json.loads((TEACH_DIR / "bc_demos.json").read_text())
        reports = [demo_audit(TEACH_DIR / d["file"]) for d in manifest["demos"]]
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


if __name__ == "__main__":
    unittest.main()
