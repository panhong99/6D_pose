"""Isaac Lab env (v3, staged curriculum): M1013 + Robotiq 2F-85 picks the 029 plate by its rim and places it on a box.

Stages (Human 2026-10-03), selected with the environment variable PP_STAGE (1..4); each stage contains the previous ones
and starts exactly at the Human home joints (-5.98, -14.75, 117.74, -3.92, 7.46, -85.07 deg, no reset noise), so a stage-k policy is initialised from the stage-(k-1) policy:
  1 reach : move collision-free to the rim grasp pose (approach from above, gripper open, tool down, fingers radial)
  2 grasp : + close on the rim, lift >= 5 cm and hold 5 s
  3 carry : + carry the plate 0.5 m sideways until it is above the target box (held, lifted) for 1 s
  4 place : + lower it onto the target box, open the gripper and retreat 5 cm, plate still for 1 s
Success of a stage ends the episode with a bonus; the curriculum moves on only at >= 95 % success (rl/curriculum.py).

Frames: world == robot base (env origin), +x toward the steel frame, +y robot-left, z up.  Table top z = -0.02.
Safety: rl/safety_shield.py rejects any arm target closer than 5 cm to the steel frame / 2 cm to the table (voxel distance
fields); contact of an arm link with anything also ends the episode with a penalty.
Plate: the scanned 029 mesh (assets/plate029.usd, origin = bottom centre), placed at the table centre (Human: far from the
frame), target box 0.5 m away in -y.  Pose observations are simulator truth + 5 mm noise (stand-in for FoundationPose).
"""
import math
import os
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from m1013_2f85_cfg import M1013_2F85_CFG
from safety_shield import SafetyShield

STAGE = int(os.environ.get("PP_STAGE", "1"))
assert STAGE in (1, 2, 3, 4), STAGE

# ---------------------------------------------------------------- geometry (m), robot base frame
TABLE_TOP_Z = -0.02
BOX_H, BOX_XY = 0.0535, 0.08                      # Human 53-54 mm; footprint PLACEHOLDER
PLATE_USD = str(Path(__file__).resolve().parent / "assets" / "plate029.usd")
PLATE_MASS = 0.30                                 # ASSUMED (set in make_plate_usd.py)
PLATE_XY = (1.10, 0.0)                            # Human 2026-10-03: moved out from (0.60, 0) - too close to the base
TARGET_XY = (1.10, -0.45)                         # 0.45 m sideways: at -0.50 the rim grasp is 4.5 cm from an upright (< 5 cm shield)
REST_Z = TABLE_TOP_Z + BOX_H                      # plate root (bottom centre) resting on a box
RIM_R, RIM_DZ = 0.128, 0.016                      # grasp on the rim wall: radius, height above the plate bottom
RIM_DIR = (-1.0, 0.0)                             # robot-facing side of the plate
FINGER_CLOSED = 0.0425
LIFT_OK = 0.05                                    # "lifted" = bottom 5 cm above its rest height
CTRL_HZ = 30                                      # policy rate (sim 120 Hz, decimation 4)
HOLD_STEPS = 150                                  # stage 2: hold 5 s (Human)
DWELL_STEPS = {1: 15, 3: 30, 4: 30}               # success must persist this many policy steps
EPISODE_S = {1: 8.0, 2: 14.0, 3: 18.0, 4: 22.0}


# ---------------------------------------------------------------- safe IK action
class SafeDiffIKAction(DifferentialInverseKinematicsAction):
    """Differential-IK TCP pose deltas whose joint targets must pass the voxel shield; a rejected target keeps the last
    safe target (the arm stops there), and the rejection is recorded for the penalty term."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.shield = SafetyShield(self.device)
        self._safe_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self.blocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        self.blocked[:] = False
        super().process_actions(actions)

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        q = self._asset.data.joint_pos[:, self._joint_ids]
        q_des = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), q)
        ok = self.shield.safe(q_des)
        self.blocked |= ~ok
        self._safe_target = torch.where(ok[:, None], q_des, self._safe_target)
        self._asset.set_joint_position_target(self._safe_target, self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._safe_target[ids] = self._asset.data.default_joint_pos[ids][:, self._joint_ids]
        self.blocked[ids] = False


@configclass
class SafeDiffIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type = SafeDiffIKAction


# ---------------------------------------------------------------- scene
def _box(name, xy, color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_TOP_Z + BOX_H / 2)),
        spawn=sim_utils.CuboidCfg(size=(BOX_XY, BOX_XY, BOX_H), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = M1013_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, TABLE_TOP_Z - 0.05)),
        spawn=sim_utils.CuboidCfg(size=(1.8, 1.2, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3)),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)))
    start_box = _box("StartBox", PLATE_XY, (0.55, 0.40, 0.25))
    target_box = _box("TargetBox", TARGET_XY, (0.30, 0.55, 0.30))
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(PLATE_XY[0], PLATE_XY[1], REST_Z)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                                                         max_depenetration_velocity=2.0)))
    # arm links + gripper body (fingers excluded): any contact ends the episode (unfiltered net force, self-collision off)
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(link_[1-6]|gripper_base)", history_length=1)
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)),
                         spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


# ---------------------------------------------------------------- task state (computed once per env step)
def _tcp(env):
    r = env.scene["robot"]
    i = r.body_names.index("tcp")
    return r.data.body_pos_w[:, i], r.data.body_quat_w[:, i]


def _axis(q, k, n):
    v = torch.zeros(n, 3, device=q.device)
    v[:, k] = 1.0
    return quat_apply(q, v)


def _const(env, xyz):
    return torch.tensor(xyz, device=env.device).unsqueeze(0) + env.scene.env_origins


def _fingers(env):
    r = env.scene["robot"]
    ids = [r.joint_names.index(n) for n in ("left_finger_joint", "right_finger_joint")]
    return r.data.joint_pos[:, ids].mean(1) / FINGER_CLOSED                        # 0 open .. 1 closed


def _state(env):
    """All task quantities, cached per env step (ManagerBasedRLEnv computes terminations before rewards)."""
    st = getattr(env, "_pp", None)
    if st is not None and st["step"] == env.common_step_counter:
        return st
    n = env.num_envs
    if st is None:
        st = {"dwell": torch.zeros(n, device=env.device), "hold": torch.zeros(n, device=env.device), "step": -1}
    tcp, tq = _tcp(env)
    plate = env.scene["plate"]
    p = plate.data.root_pos_w
    rim = p + torch.tensor([RIM_DIR[0] * RIM_R, RIM_DIR[1] * RIM_R, RIM_DZ], device=env.device)
    start_rest = _const(env, (PLATE_XY[0], PLATE_XY[1], REST_Z))
    target_rest = _const(env, (TARGET_XY[0], TARGET_XY[1], REST_Z))
    tool_z = _axis(tq, 2, n)
    finger_ax = _axis(tq, 1, n)                                                     # tcp y = finger closing axis
    radial = torch.tensor([RIM_DIR[0], RIM_DIR[1], 0.0], device=env.device).expand(n, 3)
    fingers = _fingers(env)
    d_rim = torch.norm(tcp - rim, dim=1)
    d_rim_xy = torch.norm((tcp - rim)[:, :2], dim=1)
    down = torch.clamp(-tool_z[:, 2], min=0.0)                                      # cos(tool z, -Z)
    align = torch.abs((finger_ax * radial).sum(1))                                  # |cos(finger axis, radial)|
    at_grasp = (d_rim < 0.012) & (down > math.cos(math.radians(10))) & (align > math.cos(math.radians(15)))
    # held: fingers closed on something, TCP still at rim distance from the plate centre, close in height
    d_centre_xy = torch.norm((tcp - p)[:, :2], dim=1)
    held = (fingers > 0.15) & (fingers < 0.98) & ((d_centre_xy - RIM_R).abs() < 0.03) & \
           ((tcp[:, 2] - p[:, 2] - RIM_DZ).abs() < 0.04)
    lift = p[:, 2] - start_rest[:, 2]
    lifted = held & (lift > LIFT_OK)
    d_tgt_xy = torch.norm((p - target_rest)[:, :2], dim=1)
    speed = torch.norm(plate.data.root_lin_vel_w, dim=1)
    placed = (d_tgt_xy < 0.025) & ((p[:, 2] - target_rest[:, 2]).abs() < 0.01) & (speed < 0.05)
    plate_moved = torch.norm((p - start_rest)[:, :2], dim=1) > 0.02                # knocked off its place
    released = placed & (fingers < 0.05) & (tcp[:, 2] > p[:, 2] + 0.025 + 0.05)

    if STAGE == 1:
        cond = at_grasp & (fingers < 0.05) & ~plate_moved
    elif STAGE == 3:
        cond = lifted & (d_tgt_xy < 0.03)
    else:
        cond = released
    st["dwell"] = torch.where(cond, st["dwell"] + 1, torch.zeros_like(st["dwell"]))
    st["hold"] = torch.where(lifted, st["hold"] + 1, torch.zeros_like(st["hold"]))
    success = st["hold"] >= HOLD_STEPS if STAGE == 2 else st["dwell"] >= DWELL_STEPS[STAGE]
    st.update(step=env.common_step_counter, tcp=tcp, rim=rim, p=p, lift=lift, d_rim=d_rim, d_rim_xy=d_rim_xy, down=down,
              align=align, at_grasp=at_grasp, fingers=fingers, held=held, lifted=lifted, d_tgt_xy=d_tgt_xy,
              target_rest=target_rest, placed=placed, released=released, plate_moved=plate_moved, success=success)
    env._pp = st
    return st


def reset_stage_state(env, env_ids):
    st = getattr(env, "_pp", None)
    if st is not None:
        st["dwell"][env_ids] = 0
        st["hold"][env_ids] = 0
        st["step"] = -1


# ---------------------------------------------------------------- observations
def _to_base(env, pos_w):
    r = env.scene["robot"]
    return subtract_frame_transforms(r.data.root_pos_w, r.data.root_quat_w, pos_w)[0]


def obs_tcp(env):
    return _to_base(env, _tcp(env)[0])


def obs_tool_axes(env):
    q = _tcp(env)[1]
    return torch.cat([_axis(q, 2, env.num_envs), _axis(q, 1, env.num_envs)], dim=1)


def obs_plate(env):
    return _to_base(env, env.scene["plate"].data.root_pos_w)


def obs_plate_up(env):
    return _axis(env.scene["plate"].data.root_quat_w, 2, env.num_envs)


def obs_target(env):
    return _to_base(env, _const(env, (TARGET_XY[0], TARGET_XY[1], REST_Z)))


def obs_progress(env):
    st = _state(env)
    return torch.stack([st["hold"] / HOLD_STEPS, st["dwell"] / 30.0], dim=1).clamp(max=1.0)


# ---------------------------------------------------------------- rewards (per step; Isaac Lab multiplies by dt)
def r_reach(env, std: float):
    return 1.0 - torch.tanh(_state(env)["d_rim"] / std)


def r_above_first(env):
    """Height deficit (m) below a required height while not yet over the rim: forces the approach from above."""
    st = _state(env)
    need = 0.08 * torch.clamp((st["d_rim_xy"] - 0.02) / 0.05, 0.0, 1.0)
    return torch.clamp(st["rim"][:, 2] + need - st["tcp"][:, 2], min=0.0) * (~st["held"]).float()


def r_down(env):
    return _state(env)["down"] ** 2


def r_align(env):
    st = _state(env)
    return st["align"] ** 2 * (1.0 - torch.tanh(st["d_rim"] / 0.1))


def r_open_before_grasp(env):
    st = _state(env)
    return st["fingers"] * (~st["at_grasp"] & ~st["held"]).float()


def r_close_at_grasp(env):
    st = _state(env)
    return st["fingers"] * (st["at_grasp"] | st["held"]).float()


def r_lift(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["lift"] / 0.08, 0.0, 1.0)


def r_carry(env, std: float):
    st = _state(env)
    return st["lifted"].float() * (1.0 - torch.tanh(st["d_tgt_xy"] / std))


def r_lower(env):
    st = _state(env)
    over = (st["d_tgt_xy"] < 0.03) & st["held"]
    return over.float() * (1.0 - torch.tanh((st["p"][:, 2] - st["target_rest"][:, 2]).abs() / 0.03))


def r_release(env):
    st = _state(env)
    up = torch.clamp((st["tcp"][:, 2] - st["p"][:, 2] - 0.025) / 0.075, 0.0, 1.0)
    return st["placed"].float() * (1.0 - st["fingers"]) * (0.5 + 0.5 * up)


def r_success(env):
    return _state(env)["success"].float()


def r_shield(env):
    return env.action_manager.get_term("arm_action").blocked.float()


def collision_force(env, threshold: float = 1.0):
    f = env.scene["arm_contacts"].data.net_forces_w
    return (torch.norm(f, dim=-1).amax(dim=1) > threshold).float()


def r_plate_knocked(env):
    st = _state(env)
    return (st["plate_moved"] & ~st["held"] & ~st["placed"]).float()


# ---------------------------------------------------------------- terminations
def t_success(env):
    return _state(env)["success"]


def t_collision(env):
    return collision_force(env) > 0.5


def t_plate_lost(env):
    """Stage 1: plate knocked off its place.  Later stages: plate below the box tops and not held (dropped)."""
    st = _state(env)
    if STAGE == 1:
        return st["plate_moved"]
    return (st["p"][:, 2] < REST_Z - 0.03) & ~st["held"]


# ---------------------------------------------------------------- MDP groups
@configclass
class ActionsCfg:
    # TCP pose deltas [dx, dy, dz, droll, dpitch, dyaw] x 0.03 (m / rad) per policy step, base frame, shielded
    arm_action = SafeDiffIKActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], body_name="tcp", scale=0.03,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"))
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=[".*_finger_joint"],
        open_command_expr={".*_finger_joint": 0.0}, close_command_expr={".*_finger_joint": FINGER_CLOSED})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.5)
        tcp = ObsTerm(func=obs_tcp)
        tool_axes = ObsTerm(func=obs_tool_axes)
        plate = ObsTerm(func=obs_plate, noise=Unoise(n_min=-0.005, n_max=0.005))     # FoundationPose stand-in
        plate_up = ObsTerm(func=obs_plate_up)
        target = ObsTerm(func=obs_target)
        progress = ObsTerm(func=obs_progress)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    plate_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                               params={"asset_cfg": SceneEntityCfg("plate"), "static_friction_range": (1.0, 1.0),
                                       "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    finger_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                                params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_finger"), "static_friction_range": (1.0, 1.0),
                                        "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_plate = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                            params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-math.pi, math.pi)},
                                    "velocity_range": {}, "asset_cfg": SceneEntityCfg("plate")})
    reset_state = EventTerm(func=reset_stage_state, mode="reset")


@configclass
class RewardsCfg:
    # stage 1 (always on): reach the rim from above, tool down, fingers radial, gripper open until there
    reach = RewTerm(func=r_reach, params={"std": 0.2}, weight=1.0)
    reach_fine = RewTerm(func=r_reach, params={"std": 0.02}, weight=1.5)
    above_first = RewTerm(func=r_above_first, weight=-20.0)
    down = RewTerm(func=r_down, weight=0.5)
    align = RewTerm(func=r_align, weight=0.5)
    open_before_grasp = RewTerm(func=r_open_before_grasp, weight=-1.0)
    success = RewTerm(func=r_success, weight=3000.0)          # x dt (1/30) = +100 at the success step
    # safety / smoothness (always on)
    shield = RewTerm(func=r_shield, weight=-1.0)
    collision = RewTerm(func=collision_force, weight=-1500.0)  # x dt = -50, episode ends
    knocked = RewTerm(func=r_plate_knocked, weight=-3.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=t_success)
    collision = DoneTerm(func=t_collision)
    plate_lost = DoneTerm(func=t_plate_lost)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=3.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_S[STAGE]
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024   # 16k overflowed with 4096 envs (2026-10-03 hang)
        # stage-specific terms (stage k keeps everything from the earlier stages)
        if STAGE >= 2:
            self.rewards.open_before_grasp.weight = -0.5
            self.rewards.close_at_grasp = RewTerm(func=r_close_at_grasp, weight=1.0)
            self.rewards.lift = RewTerm(func=r_lift, weight=3.0)
            self.rewards.success.weight = 6000.0          # x dt = +200
        if STAGE >= 3:
            self.rewards.carry = RewTerm(func=r_carry, params={"std": 0.25}, weight=2.0)
            self.rewards.carry_fine = RewTerm(func=r_carry, params={"std": 0.03}, weight=2.0)
        if STAGE >= 4:
            self.rewards.lower = RewTerm(func=r_lower, weight=3.0)
            self.rewards.release = RewTerm(func=r_release, weight=3.0)
            self.rewards.success.weight = 9000.0          # x dt = +300
        # steel frame (measured uprights/rails, calibration/workcell.frame_layout): physical colliders in every env
        import json
        import sys
        calib = Path(__file__).resolve().parents[1] / "calibration"
        sys.path.insert(0, str(calib))
        import workcell as wc
        cell = dict(wc.FRAME_CELL)
        cell.update({k: v for k, v in json.loads((calib / "data" / "real_cell_measured.json").read_text()).items()
                     if not k.startswith("_")})
        for name, centre, size in wc.frame_layout(cell):
            if name == "table":
                continue
            setattr(self.scene, f"Frame_{name}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Frame_{name}", init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(centre)),
                spawn=sim_utils.CuboidCfg(size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.74, 0.77)))))


@configclass
class PickPlaceEnvCfg_PLAY(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.9, 1.5, 1.3)
        self.viewer.lookat = (0.9, -0.2, 0.1)


gym.register(id="Pick-Place-M1013-2F85-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}:PickPlaceEnvCfg",
                     "rsl_rl_cfg_entry_point": "pick_place_agent:PickPlacePPORunnerCfg"})
gym.register(id="Pick-Place-M1013-2F85-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}:PickPlaceEnvCfg_PLAY",
                     "rsl_rl_cfg_entry_point": "pick_place_agent:PickPlacePPORunnerCfg"})
