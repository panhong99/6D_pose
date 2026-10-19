"""Offline M1013 FK parity check; never connects to a robot.

Run with the isolated PyRoki environment and JAX_PLATFORM_NAME=cpu.
"""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calibration"))
from robot_model import DEFAULT_URDF, Kinematics  # noqa: E402


def main():
    jax.config.update("jax_enable_x64", True)
    old = Kinematics()
    urdf = yourdfpy.URDF.load(str(DEFAULT_URDF), load_meshes=False)
    new = pk.Robot.from_urdf(urdf)
    assert list(new.joints.actuated_names) == [f"joint_{i}" for i in range(1, 7)]
    link = new.links.names.index("link_6")
    rng = np.random.default_rng(20261019)
    lo = np.array([j[3] for j in old.joints])
    hi = np.array([j[4] for j in old.joints])
    q = rng.uniform(lo, hi, size=(1000, 6))
    start = time.perf_counter()
    reference = [old.fk(x) for x in q]
    old_s = time.perf_counter() - start
    start = time.perf_counter()
    first = np.asarray(new.forward_kinematics(q[:1]))
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    poses = np.asarray(new.forward_kinematics(q))[:, link]
    poses_s = time.perf_counter() - start
    start = time.perf_counter()
    _ = np.asarray(new.forward_kinematics(q))[:, link]
    warm_s = time.perf_counter() - start
    xyz = poses[:, 4:]
    quat_xyzw = poses[:, [1, 2, 3, 0]]
    pos = np.array([np.linalg.norm(t[:3, 3] - p) for t, p in zip(reference, xyz)]) * 1000
    rot = np.array([
        np.degrees((Rotation.from_matrix(t[:3, :3]).inv() * Rotation.from_quat(r)).magnitude())
        for t, r in zip(reference, quat_xyzw)
    ])
    result = {
        "samples": len(q), "seed": 20261019,
        "urdf": str(DEFAULT_URDF), "joint_order": list(new.joints.actuated_names),
        "position_mm": {"median": float(np.median(pos)), "max": float(np.max(pos))},
        "orientation_deg": {"median": float(np.median(rot)), "max": float(np.max(rot))},
        "timing_s": {"legacy_1000": old_s, "pyroki_first_1": first_s,
                     "pyroki_batch_1000_first": poses_s, "pyroki_batch_1000_warm": warm_s},
        "note": "FK parity only; not an IK, collision, or robot-execution validation.",
    }
    output = Path(__file__).with_name("fk_result.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
