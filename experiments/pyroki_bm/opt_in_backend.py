"""Explicit offline PyRoki backend; never connected to the demo executor.

Callers supply the same held-plate and floor checkers used by the current
planner.  A carry result is returned only after their dense clearance gate
passes.  No fallback path is silently accepted.
"""
from pathlib import Path

import numpy as np

from bridge import PyrokiKinematics
from trajopt_backend import CandidateRejected, optimize, validate


class OfflinePyrokiBackend:
    def __init__(self, urdf_path: str | Path, robot_collision):
        self.kinematics = PyrokiKinematics(urdf_path)
        self.robot_collision = robot_collision

    def fk(self, joints):
        return self.kinematics.fk(joints)

    def ik(self, target, seed, **kwargs):
        return self.kinematics.ik(target, seed, **kwargs)

    def carry(self, start, goal, *, held, floor, attached_cloud, boxes,
              basket_T, basket_half_xy, floor_z, n=20, maxiter=120,
              max_wrist_deg=100.0, seed_paths=()):
        """Return ``(path, attempts)`` or raise ``CandidateRejected``.

        ``held`` and ``floor`` must be the production MultiChecker and
        FloorCheck for this exact scene, including standing plates. The caller
        retains responsibility for validating the subsequent vertical drop.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if start.shape != (6,) or goal.shape != (6,):
            raise ValueError("start and goal must each contain six joints")
        if held is None or floor is None:
            raise ValueError("held and floor checkers are required")
        path, attempts = optimize(
            self.kinematics.robot, self.robot_collision, start, goal,
            held, floor, attached_cloud, boxes, basket_T, basket_half_xy,
            floor_z, n=n, maxiter=maxiter,
            max_wrist_deg=max_wrist_deg, seed_paths=seed_paths,
        )
        # Keep the wrapper fail-closed if the optimizer's return contract
        # changes. Endpoint preservation matters for the saved drop segment.
        path = np.asarray(path, dtype=float)
        if path.shape != (n, 6) or not np.array_equal(path[0], start) or not np.array_equal(path[-1], goal):
            raise CandidateRejected([*attempts, {"reason": "endpoints_or_shape", "expected_n": n}])
        limits = (np.asarray(self.kinematics.robot.joints.lower_limits),
                  np.asarray(self.kinematics.robot.joints.upper_limits))
        good, check = validate(path, held, floor, start, max_wrist_deg, limits=limits)
        if not good:
            raise CandidateRejected([*attempts, {"reason": "final_exact_gate", **check}])
        return path, attempts
