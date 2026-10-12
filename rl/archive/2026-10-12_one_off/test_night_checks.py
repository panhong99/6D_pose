"""CPU-only regression tests for the two overnight fail-closed gates."""
import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import night_checks as checks


class NightChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model_0.pt"
        self.checkpoint.write_bytes(b"fixture")

    def log(self, tag, value, name="result.log"):
        path = self.root / name
        path.write_text(tag + " " + (value if isinstance(value, str) else json.dumps(value)) + "\n")
        return path

    def outcome(self, **changes):
        result = dict(stage=2, episodes=512, checkpoint=str(self.checkpoint), success=0.94,
                      collision=0.01, plate_lost=0.0, time_out=0.05, unfinished=0.0)
        result.update(changes)
        return result

    def dataset(self):
        out, sources = self.root / "data", self.root / "sources"
        (out / "episodes").mkdir(parents=True)
        (out / "source_snapshot").mkdir()
        sources.mkdir()
        hashes = {}
        for name in checks.SOURCES:
            content = ("source " + name).encode()
            (sources / name).write_bytes(content)
            (out / "source_snapshot" / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        entries = []
        for index, split in enumerate(("train", "validation")):
            filename = f"episodes/episode_{index:04d}.npz"
            with zipfile.ZipFile(out / filename, "w") as archive:
                for key in ("obs", "next_obs"):
                    archive.writestr(key + ".npy", self.npy((4, 46)))
            entries.append(dict(file=filename, frames=4, split=split))
        manifest = dict(status="complete", task="stage2_ready_grasp_lift_hold", requested=2, seed=3001,
                        phase_switching="state-based per env", dart_noise=0.0, slow_zone_m=0.02,
                        sources_sha256=hashes, episodes=entries, total_frames=8)
        (out / "manifest.json").write_text(json.dumps(manifest))
        report = dict(all_passed=True, episodes=2, frames=8, train=1, validation=1)
        return out, sources, manifest, self.log("VERIFY_RESULT", report, "verify.log")

    @staticmethod
    def npy(shape):
        header = repr(dict(descr="<f4", fortran_order=False, shape=shape)).encode() + b"\n"
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header

    def test_valid_gate_and_cli_json(self):
        log = self.log("EVAL_JSON", self.outcome())
        result = checks.gate_check(log, self.checkpoint)
        self.assertTrue(result["qualified"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(checks.main(["gate", str(log), "--checkpoint", str(self.checkpoint)]), 0)
        self.assertTrue(json.loads(output.getvalue())["qualified"])

    def test_zero_success_full_collision_does_not_qualify(self):
        log = self.log("EVAL_JSON", self.outcome(success=0.0, collision=1.0, time_out=0.0))
        with self.assertRaises(checks.CheckError):
            checks.gate_check(log, self.checkpoint)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(checks.main(["gate", str(log), "--checkpoint", str(self.checkpoint)]), 1)
        self.assertIn("qualification", output.getvalue())

    def test_gate_rejects_invalid_metrics_identity_and_incomplete_eval(self):
        bad = [dict(success=float("nan")), dict(collision=0.03, time_out=0.03),
               dict(unfinished=0.01, time_out=0.04), dict(episodes=256), dict(stage=3),
               dict(checkpoint=str(self.root / "other.pt")), dict(success=1.1),
               dict(success=True), dict(time_out=0.0)]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(checks.CheckError):
                checks.gate_check(self.log("EVAL_JSON", self.outcome(**change)), self.checkpoint)

    def test_missing_or_duplicate_result_is_rejected(self):
        log = self.root / "missing.log"
        log.write_text("process failed\n")
        with self.assertRaises(checks.CheckError):
            checks.gate_check(log, self.checkpoint)
        log = self.log("EVAL_JSON", self.outcome())
        log.write_text(log.read_text() * 2)
        with self.assertRaises(checks.CheckError):
            checks.gate_check(log, self.checkpoint)

    def test_checkpoint_requires_one_existing_file(self):
        log = self.log("BC_CHECKPOINT", str(self.checkpoint))
        self.assertEqual(checks.checkpoint_check(log)["checkpoint"], str(self.checkpoint))
        log.write_text(log.read_text() * 2)
        with self.assertRaises(checks.CheckError):
            checks.checkpoint_check(log)
        with self.assertRaises(checks.CheckError):
            checks.checkpoint_check(self.log("BC_CHECKPOINT", str(self.root / "missing.pt")))

    def test_selection_uses_success_then_lower_collision(self):
        paths = []
        for index, result in enumerate((self.outcome(success=0.95, collision=0.01, time_out=0.04),
                                        self.outcome(success=0.95, collision=0.0, time_out=0.05),
                                        self.outcome(success=0.94, collision=0.0, time_out=0.06))):
            qualified = checks.validate_gate(result, self.checkpoint, 512, 0.85, 0.02)
            qualified["seed"] = index
            path = self.root / f"candidate_{index}.json"
            path.write_text(json.dumps(qualified))
            paths.append(path)
        selected = checks.select_check(paths, self.root / "selected.json")
        self.assertEqual(selected["seed"], 1)
        self.assertEqual(json.loads((self.root / "selected.json").read_text())["seed"], 1)
        with self.assertRaises(checks.CheckError):
            checks.select_check([], self.root / "empty.json")
        paths[0].write_text(json.dumps({**selected, "qualified": False}))
        with self.assertRaises(checks.CheckError):
            checks.select_check(paths, self.root / "invalid.json")

    def test_complete_dataset_and_fresh_verification(self):
        out, sources, _, verify = self.dataset()
        result = checks.data_check(out, 3001, 0.0, 2, verify, source_dir=sources)
        self.assertTrue(result["verified"])
        self.assertEqual(result["obs_dim"], 46)
        # An old verification.json cannot compensate for a failed current verify log.
        (out / "verification.json").write_text(json.dumps({"all_passed": True}))
        verify.write_text("Traceback: invalid episode\n")
        with self.assertRaises(checks.CheckError):
            checks.data_check(out, 3001, 0.0, 2, verify, source_dir=sources)

    def test_incomplete_count_seed_dart_or_sources_rejected(self):
        out, sources, manifest, verify = self.dataset()
        for change in (dict(status="incomplete"), dict(requested=1), dict(seed=3002),
                       dict(dart_noise=0.2), dict(slow_zone_m=0.0), dict(total_frames=7),
                       dict(sources_sha256={})):
            (out / "manifest.json").write_text(json.dumps({**manifest, **change}))
            with self.subTest(change=change), self.assertRaises(checks.CheckError):
                checks.data_check(out, 3001, 0.0, 2, verify, source_dir=sources)

    def test_wrong_observation_dim_or_verification_counts_rejected(self):
        out, sources, _, verify = self.dataset()
        with self.assertRaises(checks.CheckError):
            checks.data_check(out, 3001, 0.0, 2, verify, obs_dim=43, source_dir=sources)
        verify.write_text('VERIFY_RESULT {"all_passed":true,"episodes":2,"frames":7,"train":1,"validation":1}\n')
        with self.assertRaises(checks.CheckError):
            checks.data_check(out, 3001, 0.0, 2, verify, source_dir=sources)


if __name__ == "__main__":
    unittest.main()
