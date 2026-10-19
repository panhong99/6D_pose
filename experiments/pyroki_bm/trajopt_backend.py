"""Opt-in PyRoki FK based carry optimizer with a mandatory legacy safety gate.

This offline candidate generator never commands a robot. All accepted segments
must pass the current held-plate and floor checkers before they are returned.
"""
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from scipy.optimize import minimize


class CandidateRejected(RuntimeError):
    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__("PyRoki candidates failed the legacy clearance gate")


def _point_box_distance(points, centres, rotations, half):
    # points [time, point, xyz], boxes [box, xyz]
    local = jnp.einsum("tpbi,bij->tpbj", points[:, :, None, :] - centres[None, None], rotations)
    delta = jnp.abs(local) - half[None, None]
    outside = jnp.maximum(delta, 0)
    return jnp.sqrt(jnp.sum(outside * outside, axis=-1) + 1e-12) + jnp.minimum(jnp.max(delta, axis=-1), 0)


def _densify(path, step_deg=2.0):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.degrees(np.max(np.abs(b - a))) / step_deg)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.asarray(out)


def _resample_path(path, n):
    path = np.asarray(path, float)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-12]
    if distance[-1] <= 1e-12:
        return np.repeat(path[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0, distance[-1], n), distance[keep], path[keep, i])
                     for i in range(path.shape[1])], axis=1)


def validate(path, held, floor, start, max_wrist_deg=100.0, step_deg=2.0, limits=None):
    """Exact existing checkers are authoritative, including held plate and floor."""
    dense = _densify(np.asarray(path, float), step_deg)
    if not np.isfinite(dense).all():
        return False, {"reason": "nonfinite"}
    if limits is not None:
        lo, hi = limits
        if np.any(dense < np.asarray(lo)) or np.any(dense > np.asarray(hi)):
            return False, {"reason": "joint_limits"}
    wrist = np.max(np.degrees(np.abs(dense[:, [3, 5]] - np.asarray(start)[[3, 5]])))
    if wrist > max_wrist_deg:
        return False, {"reason": "wrist", "max_wrist_deg": float(wrist)}
    low_held = min(float(held.clearance(q)) for q in dense)
    low_floor = min(float(floor.clearance(q)) for q in dense)
    return low_held >= 1 and low_floor >= 1, {
        "min_held_clearance": low_held, "min_floor_clearance": low_floor,
        "max_wrist_deg": float(wrist), "checked_points": len(dense),
    }


def optimize(robot, robot_collision, start, goal, held, floor, attached_cloud,
             boxes, basket_T, basket_half_xy, floor_z, n=20,
             max_wrist_deg=100.0, maxiter=120, seed_paths=(), robot_penalty_weight=0.0):
    """Return a checked path or raise RuntimeError.

    `boxes` has the exact `plate_plans.basket_boxes` tuple format. This optimizer
    adds differentiable held-cloud/floor penalties and can optionally add a
    robot-capsule penalty. The capsule proxy disagreed with the existing
    checker on valid saved paths, so its default weight is zero. Final
    acceptance always uses the existing collision/floor contract.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    cloud = jnp.asarray(attached_cloud, dtype=jnp.float32)
    centres = jnp.asarray([b[1] for b in boxes], dtype=jnp.float32)
    rotations = jnp.asarray([b[3] for b in boxes], dtype=jnp.float32)
    half = jnp.asarray([b[2] for b in boxes], dtype=jnp.float32) / 2
    # Box orientation is converted from the supplied rotation matrices.
    from scipy.spatial.transform import Rotation
    box_wxyz = Rotation.from_matrix(np.asarray(rotations)).as_quat()[:, [3, 0, 1, 2]]
    from pyroki import collision as coll
    world = coll.Box.from_extent(np.asarray(half) * 2, np.asarray(centres), box_wxyz)
    inv = np.linalg.inv(np.asarray(basket_T))
    basket_R = jnp.asarray(inv[:3, :3], dtype=jnp.float32)
    basket_p = jnp.asarray(inv[:3, 3], dtype=jnp.float32)
    basket_half = jnp.asarray(basket_half_xy, dtype=jnp.float32)
    start_j, goal_j = jnp.asarray(start), jnp.asarray(goal)
    limit_lo = jnp.asarray(robot.joints.lower_limits)
    limit_hi = jnp.asarray(robot.joints.upper_limits)
    link = robot.links.names.index("link_6")

    @jax.jit
    def value(flat):
        path = jnp.concatenate([start_j[None], flat.reshape((n - 2, 6)), goal_j[None]], axis=0)
        pose = robot.forward_kinematics(path)[:, link]
        R = jaxlie.SO3(pose[:, :4]).as_matrix()
        points = jnp.einsum("tij,pj->tpi", R, cloud) + pose[:, None, 4:]
        box_dist = _point_box_distance(points, centres, rotations, half)
        plate_pen = jnp.sum(jnp.square(jnp.maximum(0.04 - box_dist, 0)))
        if robot_penalty_weight:
            caps = robot_collision.at_config(robot, path)
            # The fixed base has no moving collision sample in the legacy checker.
            caps = jax.tree.map(lambda x: x[:, 1:], caps)
            cap_dist = coll.collide(caps.reshape((n, robot_collision.num_links - 1, 1)),
                                     world.reshape((1, 1, len(boxes))))
            cap_dist = jnp.nan_to_num(cap_dist, nan=1.0, posinf=1.0, neginf=-1.0)
            robot_pen = jnp.sum(jnp.square(jnp.maximum(0.04 - cap_dist, 0)))
        else:
            robot_pen = jnp.asarray(0.0)
        local = jnp.einsum("tpi,ij->tpj", points, basket_R.T) + basket_p
        over = jnp.all(jnp.abs(local[:, :, :2]) <= basket_half, axis=-1)
        floor_pen = jnp.sum(jnp.square(jnp.maximum(floor_z + 0.01 - points[:, :, 2], 0)) * over)
        step = jnp.diff(path, axis=0)
        accel = jnp.diff(path, n=2, axis=0)
        wrist = jnp.maximum(jnp.abs(path[:, jnp.array([3, 5])] - start_j[jnp.array([3, 5])]) - jnp.deg2rad(max_wrist_deg), 0)
        limits = jnp.maximum(limit_lo - path, 0) + jnp.maximum(path - limit_hi, 0)
        smooth = 0.05 * jnp.sum(step * step) + jnp.sum(accel * accel)
        penalties = dict(plate=plate_pen, robot=robot_pen, floor=floor_pen,
                         wrist=jnp.sum(wrist * wrist), limits=jnp.sum(limits * limits))
        total = (smooth + 2e4 * (plate_pen + floor_pen) + robot_penalty_weight * robot_pen +
                 1e4 * (penalties["wrist"] + penalties["limits"]))
        return total, dict(smooth=smooth, **penalties)

    fg = jax.jit(jax.value_and_grad(value, has_aux=True))
    seeds = [np.linspace(start, goal, n)]
    # Retain the exact chosen joint endpoints. Two deterministic elbow detours
    # help the optimizer avoid a straight-line local minimum.
    for sign in (-1, 1):
        path = np.linspace(start, goal, n)
        path[:, 1] += sign * np.sin(np.linspace(0, np.pi, n)) * np.deg2rad(15)
        seeds.append(path)
    seed_names = ["linear", "elbow_minus", "elbow_plus"]
    for i, path in enumerate(seed_paths):
        path = np.asarray(path, float)
        if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
            raise ValueError("seed_path must be an N x 6 joint path")
        if not np.allclose(path[0], start) or not np.allclose(path[-1], goal):
            raise ValueError("seed_path endpoints must match start and goal")
        seeds.append(_resample_path(path, n))
        seed_names.append(f"external_{i}")
    attempts = []
    best = None
    for seed_name, seed in zip(seed_names, seeds):
        tic = time.perf_counter()
        initial_good, initial_check = validate(seed, held, floor, start, max_wrist_deg,
                                               limits=(np.asarray(limit_lo), np.asarray(limit_hi)))
        (initial_value, initial_terms), _ = fg(jnp.asarray(seed[1:-1].ravel(), dtype=jnp.float32))
        def fun(flat):
            (v, _), g = fg(jnp.asarray(flat, dtype=jnp.float32))
            vv, gg = float(v), np.asarray(g, dtype=float)
            # A degenerate capsule occasionally gives a nonfinite derivative.
            # Reject such an iterate instead of handing NaN to L-BFGS-B.
            if not np.isfinite(vv) or not np.isfinite(gg).all():
                return 1e12, np.zeros_like(np.asarray(flat, dtype=float))
            return vv, gg
        fit = minimize(fun, seed[1:-1].ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-8})
        candidate = np.vstack([start, fit.x.reshape((n - 2, 6)), goal])
        (_, final_terms), _ = fg(jnp.asarray(fit.x, dtype=jnp.float32))
        # Distinguish a knot violation from an interpolation violation when the
        # differentiable cost (evaluated at knots) and the dense gate disagree.
        knot_floor = min(float(floor.clearance(q)) for q in candidate)
        knot_held = min(float(held.clearance(q)) for q in candidate)
        good, check = validate(candidate, held, floor, start, max_wrist_deg,
                               limits=(np.asarray(limit_lo), np.asarray(limit_hi)))
        terms = {name: float(value_) for name, value_ in final_terms.items()}
        initial_terms = {name: float(value_) for name, value_ in initial_terms.items()}
        attempts.append({"seed": seed_name, "time_s": time.perf_counter() - tic,
                         "objective": float(fit.fun),
                         "initial_objective": float(initial_value),
                         "initial_cost_terms_unweighted": initial_terms,
                         "initial_valid": initial_good, "initial_gate": initial_check,
                         "final_cost_terms_unweighted": terms,
                         "min_floor_knot_clearance": knot_floor,
                         "min_held_knot_clearance": knot_held,
                         "movement_from_seed_deg": float(np.degrees(np.linalg.norm(candidate - seed))),
                         "optimizer_success": bool(fit.success), "valid": good, **check})
        if good:
            travel = float(np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum())
            if best is None or travel < best[0]:
                best = (travel, candidate)
    if best is not None:
        return best[1], attempts
    raise CandidateRejected(attempts)
