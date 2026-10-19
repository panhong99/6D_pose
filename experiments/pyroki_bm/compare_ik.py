"""Offline reachable-pose IK smoke benchmark; no collision claims."""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "calibration"), "/tmp/pyroki-bm-20261019/examples"]
from robot_model import DEFAULT_URDF, Kinematics  # noqa: E402
from pyroki_snippets._solve_ik import _solve_ik_jax  # noqa: E402


def error(fk, goal):
    return (float(np.linalg.norm(fk[:3, 3] - goal[:3, 3]) * 1000),
            float(np.degrees((Rotation.from_matrix(goal[:3, :3]).inv() *
                              Rotation.from_matrix(fk[:3, :3])).magnitude())))


def main():
    jax.config.update("jax_enable_x64", True)
    old = Kinematics()
    u = yourdfpy.URDF.load(str(DEFAULT_URDF), load_meshes=False)
    robot = pk.Robot.from_urdf(u)
    rng = np.random.default_rng(20261019)
    lo, hi = np.array([x[3] for x in old.joints]), np.array([x[4] for x in old.joints])
    targets = rng.uniform(lo + .2, hi - .2, size=(20, 6))
    seed = (lo + hi) / 2
    rows = []
    for q in targets:
        target = old.fk(q)
        quat = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        start = time.perf_counter()
        try:
            solution = old.ik(target, seed)
            legacy = {"success": True, "error": error(old.fk(solution), target)}
        except RuntimeError:
            legacy = {"success": False}
        legacy["seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        try:
            # Upstream wrapper builds a default int64 link index when x64 is on,
            # but the analytic Jacobian requires int32. Call its JIT with int32.
            solution = np.asarray(_solve_ik_jax(robot, jnp.array(robot.links.names.index("link_6"), dtype=jnp.int32),
                                                jnp.array(quat), jnp.array(target[:3, 3])))
            ep, er = error(old.fk(solution), target)
            modern = {"success": ep <= .1 and er <= .1, "error": [ep, er]}
        except Exception as exc:
            modern = {"success": False, "failure": type(exc).__name__ + ": " + str(exc)[:200]}
        modern["seconds"] = time.perf_counter() - start
        rows.append({"legacy": legacy, "pyroki": modern})
        print(len(rows), legacy["success"], modern["success"], flush=True)
    summary = {name: {"success": sum(row[name]["success"] for row in rows),
                      "median_seconds": float(np.median([row[name]["seconds"] for row in rows])),
                      "first_seconds": rows[0][name]["seconds"]}
               for name in ("legacy", "pyroki")}
    result = {"samples": len(rows), "seed": 20261019, "summary": summary,
              "rows": rows,
              "limitations": "20 random reachable FK targets; both solvers start at joint-limit midpoint; no collisions, no seed continuation, PyRoki uses upstream example pose/limit costs. Legacy success uses its internal weighted residual threshold; PyRoki success requires <=0.1 mm and <=0.1 degree."}
    Path(__file__).with_name("ik_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
