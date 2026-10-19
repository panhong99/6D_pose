"""Compare an offline PyRoki carry candidate with one stored carry prefix.

The saved plan does not record the RRT drop-start index or camera calibration
path. We infer the last floor-valid waypoint and use the latest camera pose;
the report marks these limitations and never claims an exact historical replay.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyroki as pk
import yourdfpy

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "calibration"), str(Path(__file__).parent)]
from calib_core import latest_t_base_cam, load_json  # noqa: E402
from plate_plans import C, FloorCheck, basket_boxes, plate_cloud, quadrant_checkers  # noqa: E402
from robot_model import DEFAULT_URDF, Kinematics  # noqa: E402
from trajopt_backend import CandidateRejected, optimize, validate  # noqa: E402


def metrics(path, held, floor, start):
    good, checks = validate(path, held, floor, start)
    diff = np.diff(path, axis=0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(diff, axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-12]
    sampled = np.stack([np.interp(np.linspace(0, s[-1], 101), s[keep], path[keep, i])
                        for i in range(6)], axis=1)
    acc = np.diff(sampled, n=2, axis=0)
    jerk = np.diff(sampled, n=3, axis=0)
    return dict(valid=good, **checks,
                joint_travel_deg=float(np.degrees(np.linalg.norm(diff, axis=1)).sum()),
                acceleration_l2_101=float(np.linalg.norm(acc)),
                jerk_l2_101=float(np.linalg.norm(jerk)),
                points=len(path))


def resample(path, n=101):
    path = np.asarray(path, float)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-12]
    if distance[-1] <= 1e-12:
        return np.repeat(path[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0, distance[-1], n), distance[keep], path[keep, i])
                     for i in range(path.shape[1])], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat identical solve in this process to measure warm solve and path variation")
    parser.add_argument("--library", type=Path,
                        help="Saved carry-library JSON with source plan and exact drop_start")
    parser.add_argument("--seed-from-legacy", action="store_true",
                        help="Also optimize a saved RRT seed (hybrid postprocessing, not an RRT replacement)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standing", action="append", default=[], metavar="Q:PIN",
                        help="Existing upright plate, e.g. Q4:11; may be repeated")
    a = parser.parse_args()
    if a.repeat < 1:
        parser.error("--repeat must be >= 1")
    source = json.loads(a.plan.read_text())
    if not source["quadrant_plan"]["carry"].startswith(("rrt-floor", "prior")):
        raise ValueError("Saved plan is not an RRT/prior carry")
    q = {**C["plate_rl"], **C["plate_ik_bc"]}
    basket = np.asarray(source["basket_T"], float)
    plates = []
    for item in a.standing:
        quadrant, pin = item.split(":", 1)
        pin = int(pin)
        if quadrant not in ("Q1", "Q2", "Q3", "Q4") or pin < 1:
            parser.error(f"invalid --standing {item!r}; expected Q1..Q4:PIN")
        plates.append((quadrant, pin - 1))
    H = np.asarray(source["plate_in_tcp"], float)
    tool = np.eye(4)
    tool[2, 3] = q.get("tool_length_m", 0.165)
    Tbc_path = latest_t_base_cam()
    Tbc = np.asarray(load_json(Tbc_path)["T_base_cam"], float)
    held, _, _ = quadrant_checkers(Tbc, q, basket, H, tool, plates=plates)
    cloud = plate_cloud(H, tool)
    boxes_basket = basket_boxes(basket, plates=plates)
    markers = [c[2] + s[2] / 2 for name, c, s, _ in boxes_basket if name.startswith("marker_")]
    z_floor = max(markers) + q.get("basket_carry_clear_m", 0.05)
    model_quadrant = source.get("quadrant") or source.get("quadrant_plan", {}).get("quadrant")
    if plates and model_quadrant in q.get("basket_carry_rrt_over_plates", ()):
        standing_tops = [c[2] + s[2] / 2 for name, c, s, _ in boxes_basket if name.startswith("plate_")]
        z_floor = max(z_floor, max(standing_tops) + q.get("basket_carry_plate_clear_m", 0.05))
    if plates:
        z_floor += (q.get("basket_carry_rrt_extra_m", {}) or {}).get(model_quadrant, 0.0)
    floor = FloorCheck(Kinematics(), cloud, basket, z_floor)
    saved = np.radians(np.asarray(source["legs"]["to_above"]["waypoints_deg"], float))
    if a.library is not None:
        library = json.loads(a.library.read_text())
        if Path(library["source"]).resolve() != a.plan.resolve():
            parser.error("--library source does not match plan")
        if not np.array_equal(np.asarray(library["waypoints_deg"], float),
                              np.asarray(source["legs"]["to_above"]["waypoints_deg"], float)):
            parser.error("--library waypoints do not match plan")
        k = int(library["drop_start"])
        if not 0 < k < len(saved):
            parser.error("--library drop_start is outside the carry path")
    else:
        floor_ok = [floor.clearance(x) >= 1 for x in saved]
        if not any(floor_ok):
            parser.error("no floor-valid saved waypoint under reconstructed conditions")
        k = max(i for i, valid in enumerate(floor_ok) if valid)
    old = saved[:k + 1]
    world_boxes = [b for ch in held.checkers for b in ch.boxes]
    world_boxes = [(n, c, s, np.eye(3) if len(b) == 3 else b[3])
                   for b in world_boxes for n, c, s in [b[:3]]]
    urdf = yourdfpy.URDF.load(str(DEFAULT_URDF), load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    import basket_env as be
    half = np.asarray(be.B["full"]["size_xy"], float) / 2 + 0.08
    repeats = []
    accepted_paths = []
    for _ in range(a.repeat):
        start = time.perf_counter()
        try:
            path, attempts = optimize(robot, collision, old[0], old[-1], held, floor,
                                      cloud, world_boxes, basket, half, z_floor,
                                      n=a.n, maxiter=a.maxiter,
                                      seed_paths=(old,) if a.seed_from_legacy else ())
            candidate = metrics(path, held, floor, old[0])
            accepted_paths.append(path)
        except CandidateRejected as exc:
            attempts, candidate = exc.attempts, {"valid": False, "failure": str(exc)}
        repeats.append({"wall_s": time.perf_counter() - start, "pyroki": candidate,
                        "attempts": attempts})
    variation = None
    if len(accepted_paths) > 1:
        reference = resample(accepted_paths[0])
        variation = [float(np.degrees(np.max(np.abs(resample(p) - reference))))
                     for p in accepted_paths[1:]]
    report = {
        "plan": str(a.plan), "source_carry": source["quadrant_plan"]["carry"],
        "camera_calibration_used": str(Tbc_path), "drop_start_index": k,
        "drop_start_source": str(a.library) if a.library else "last_floor_valid_inferred",
        "world_boxes": len(world_boxes), "plate_cloud_points": len(cloud),
        "standing_plates": a.standing, "model_quadrant": model_quadrant,
        "seed_from_legacy": a.seed_from_legacy,
        "floor_z_m": float(z_floor), "legacy": metrics(old, held, floor, old[0]),
        "pyroki": repeats[0]["pyroki"], "attempts": repeats[0]["attempts"],
        "pyroki_wall_s": repeats[0]["wall_s"],
        "repeats": repeats, "accepted_path_variation_max_joint_deg": variation,
        "limitations": "Historical camera calibration absent from plan; latest camera used. Current config may differ. Standing plates are operator-specified. No robot execution.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
