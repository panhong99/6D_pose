"""Plan-only IK probe for a plate standing nearly vertical.

This deliberately does not collision-check or execute.  It answers whether a fresh
FoundationPose result can be turned into a smooth top-rim approach for the M1013.
"""
import json
import time
from pathlib import Path

import numpy as np

from robot_model import HOME_JOINTS_DEG, Kinematics


def plan_standing_grasp(T_base_plate, mesh_bounds, out_dir, approach_m=0.10,
                        grasp_inset_m=0.015, tool_length_m=0.165, samples=21):
    T = np.asarray(T_base_plate, float)
    lo, hi = (np.asarray(v, float) for v in mesh_bounds)
    centre = (T @ np.r_[(lo + hi) / 2.0, 1.0])[:3]
    normal = T[:3, 2] / np.linalg.norm(T[:3, 2])
    up = np.array([0.0, 0.0, 1.0])
    top = up - normal * float(normal @ up)
    if np.linalg.norm(top) < 0.2:
        raise RuntimeError('standing grasp needs a near-vertical plate normal')
    top /= np.linalg.norm(top)
    radius = float(max((hi - lo)[:2]) / 2.0)
    tcp_grasp = centre + top * (radius - grasp_inset_m)
    tool_z = -top                         # approach from above toward the centre
    kin, home = Kinematics(), np.radians(HOME_JOINTS_DEG)
    candidates = []
    for sign in (1.0, -1.0):             # both plate faces
        x = sign * normal
        y = np.cross(tool_z, x); y /= np.linalg.norm(y)
        x = np.cross(y, tool_z); x /= np.linalg.norm(x)
        base_R = np.stack([x, y, tool_z], axis=1)
        for roll_deg in (0, 90, 180, 270):
            a = np.radians(roll_deg)
            roll = np.array([[np.cos(a), -np.sin(a), 0.0],
                             [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
            R = base_R @ roll
            seed, waypoints, errors = home, [], []
            try:
                for alpha in np.linspace(0.0, 1.0, samples):
                    tcp = tcp_grasp - tool_z * approach_m * (1.0 - alpha)
                    flange = np.eye(4)
                    flange[:3, :3] = R
                    flange[:3, 3] = tcp - tool_z * tool_length_m
                    seed = kin.ik(flange, seed)
                    waypoints.append(np.degrees(seed))
                    errors.append(float(np.linalg.norm(kin.fk(seed)[:3, 3] - flange[:3, 3])))
            except RuntimeError as exc:
                candidates.append(dict(face=int(sign), roll_deg=roll_deg, accepted=False, reason=str(exc)))
                continue
            W = np.asarray(waypoints)
            candidates.append(dict(face=int(sign), roll_deg=roll_deg, accepted=True,
                                   joint_waypoints_deg=np.round(W, 5).tolist(),
                                   max_ik_position_error_m=max(errors),
                                   max_step_deg=float(np.max(np.abs(np.diff(W, axis=0))))))
    good = [c for c in candidates if c['accepted']]
    if not good:
        raise RuntimeError('standing plate top-rim approach has no IK solution')
    best = min(good, key=lambda c: abs(c['joint_waypoints_deg'][-1][5]))
    direct = []
    q0, q1 = home, np.radians(best['joint_waypoints_deg'][0])
    n = max(1, int(np.ceil(np.max(np.abs(np.degrees(q1 - q0))) / 20.0)))
    for k in range(n + 1):
        direct.append(np.degrees(q0 + (q1 - q0) * k / n).tolist())
    approach = best['joint_waypoints_deg']
    result = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), accepted=True,
                  mode='standing_grasp_check', execute_allowed=False,
                  collision_checked=False, rrt_used=False,
                  T_base_plate=T.tolist(), plate_center_m=centre.tolist(),
                  plate_normal=normal.tolist(), tcp_grasp_m=tcp_grasp.tolist(),
                  chosen=dict(face=best['face'], roll_deg=best['roll_deg']),
                  candidates=candidates,
                  legs=dict(to_grasp=dict(planner='standing Cartesian IK, no collision check',
                                           waypoints_deg=direct + approach[1:]),
                            retreat=dict(planner='reverse of standing approach',
                                         waypoints_deg=list(reversed(approach)))))
    out = Path(out_dir) / 'standing_grasp_check.json'
    out.write_text(json.dumps(result, indent=2))
    return out, result
