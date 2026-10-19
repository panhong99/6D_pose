"""Opt-in, offline PyRoki kinematics bridge for the M1013 URDF.

This is deliberately outside the production calibration package. It provides
FK and seeded IK parity experiments, not a validated carry planner.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation


@jdc.jit
def _seeded_ik(robot, target_index, wxyz, position, seed):
    var = robot.joint_var_cls(0)
    target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), position)
    costs = [pk.costs.pose_cost_analytic_jac(robot, var, target, target_index,
                                            pos_weight=50.0, ori_weight=10.0),
             pk.costs.limit_constraint(robot, var)]
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[var]).analyze()
    initial = jaxls.VarValues.make((var.with_value(seed),))
    return problem.solve(initial_vals=initial, verbose=False,
                         linear_solver="dense_cholesky",
                         trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))[var]


class PyrokiKinematics:
    def __init__(self, urdf_path: str | Path):
        urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)
        self.robot = pk.Robot.from_urdf(urdf)
        self.link = self.robot.links.names.index("link_6")
        if self.robot.joints.actuated_names != tuple(f"joint_{i}" for i in range(1, 7)):
            raise ValueError("Unexpected M1013 joint order")

    def fk(self, joints):
        pose = np.asarray(self.robot.forward_kinematics(jnp.asarray(joints)))[self.link]
        out = np.eye(4)
        out[:3, :3] = Rotation.from_quat(pose[[1, 2, 3, 0]]).as_matrix()
        out[:3, 3] = pose[4:]
        return out

    def ik(self, target, seed, position_tol_m=1e-4, angle_tol_rad=1e-3):
        quat = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        result = np.asarray(_seeded_ik(self.robot, jnp.array(self.link, dtype=jnp.int32),
                                       jnp.asarray(quat), jnp.asarray(target[:3, 3]),
                                       jnp.asarray(seed)))
        actual = self.fk(result)
        p = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        r = (Rotation.from_matrix(target[:3, :3]).inv() *
             Rotation.from_matrix(actual[:3, :3])).magnitude()
        if not np.isfinite(result).all() or p > position_tol_m or r > angle_tol_rad:
            raise RuntimeError(f"PyRoki IK residual p={p:.6g} m, r={r:.6g} rad")
        return result
